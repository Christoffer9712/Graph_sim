import networkx as nx
import numpy as np
from dataclasses import dataclass
from enum import Enum
import astropy.units as unit
from astropy.time import TimeDelta
from ground_network.topology import C_VACUUM
from ground_network.topology import C_VACUUM
from utils.coordinates import eci_to_latlon, eci_to_latlon_batch

# ──────────────────────────────────────────────────────────────
# Link parameter tables  (Tables 1 & 2)
# All stored values are plain floats: distances in km, times in minutes.
# ──────────────────────────────────────────────────────────────

ALWAYS_UP = True  # If True, links never go DOWN (for testing visualization without stochasticity)

class LinkType(Enum):
    INTRA_PLANE_ISL = "intra_plane_isl"
    INTER_PLANE_ISL = "inter_plane_isl"


@dataclass(frozen=True)
class LinkParams:
    # --- Availability (Table 1) ---
    mttf_ref:   float   # minutes
    d_ref:      float   # km
    mttr:       float   # minutes
    gamma_fail: float
    # --- PER (Table 2) ---
    per_ref:    float
    gamma_per:  float
    alpha_load: float
    cv:         float

# Dimensions are s and km
LINK_PARAMS: dict[LinkType, LinkParams] = {
    LinkType.INTRA_PLANE_ISL: LinkParams(
        mttf_ref=100000, d_ref=2400, mttr=500,  gamma_fail=0.5,
        per_ref=1e-5,    gamma_per=2.0, alpha_load=0.5, cv=0.1,
    ),
    LinkType.INTER_PLANE_ISL: LinkParams(
        mttf_ref=300000,  d_ref=1000, mttr=1000, gamma_fail=1.0,
        per_ref=1e-4,    gamma_per=2.0, alpha_load=0.5, cv=0.2,
    ),
}

def _mttf(params: LinkParams, distance_km: float) -> float:
    """Distance-dependent MTTF in minutes (Eq. 2)."""
    return params.mttf_ref * (params.d_ref / distance_km) ** params.gamma_fail

def _markov_step(state: str, params: LinkParams,
                 distance_km: float, dt_sec: float) -> str:
    """
    One Markov step for a link's UP/DOWN state (Eq. 1).
    All inputs are plain floats.
    """
    
    if state == 'UP':
        mttf   = _mttf(params, distance_km)            
        p_fail = 1.0 - np.exp(-dt_sec / mttf)          
        return 'DOWN' if np.random.random() < p_fail else 'UP'
    
    p_rec  = 1.0 - np.exp(-dt_sec / params.mttr)
    return 'UP' if np.random.random() < p_rec else 'DOWN'

def _sample_per(params: LinkParams, distance_km: float,
                utilization: float = 0.0):
    """
    Sample PER ~ Beta(α, β) (Eqs. 3–5).
    All inputs are plain floats.
    """
    mu = (params.per_ref
          * (distance_km / params.d_ref) ** params.gamma_per
          * (1.0 + params.alpha_load * utilization))
    mu = float(np.clip(mu, 0.0, 1.0))

    if mu <= 0.0:
        return 0.0
    if mu >= 1.0:
        return 1.0

    var = (params.cv * mu) ** 2
    var = min(var, 0.999 * mu * (1.0 - mu))   # keep Beta params positive

    conc = mu * (1.0 - mu) / var - 1.0
    return (np.random.beta(mu * conc, (1.0 - mu) * conc))

class SatelliteNetwork:
    """
    Maintains a NetworkX graph of the Walker-Delta constellation.

    Positions are stored internally as plain numpy arrays in km so that
    no Astropy Quantity ever enters the graph or the link models.

    Edge attributes
    ───────────────
    distance  (float km)
    link_type (LinkType)
    state     ('UP' | 'DOWN')
    per       (float)
    """
    def __init__(self, satellites, max_distance_km,
                 P: int, S: int, F: int, start_time, only_europe=False):
        # Strip units from max_distance immediately
        self.max_distance_km = float(max_distance_km / unit.km)
        self.satellites      = satellites
        self.P = P
        self.S = S
        self.F = F
        self.only_europe = only_europe
        self.start_time = start_time
        self._id_to_sat: dict[str, object] = {}
        self._plane_to_sat: dict[tuple, object] = {
            (s.plane, s.index): s for s in satellites
        }

        self.graph = nx.Graph()
        self.euro_graph = nx.Graph()
        self._euro_nodes: set[str] = set()
        self.nbrUpdates = 0
        self._init_nodes(self.start_time)
        self._init_edges()

    # ──────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────

    def update(self, dt) -> tuple[nx.Graph, nx.Graph]:
        """
        Incremental update for one simulation step.

        Parameters
        ──────────
        dt : step duration — Astropy Quantity (any time unit) or plain
             number interpreted as seconds.
        """
        self.nbrUpdates += 1
        self._refresh_positions(self.start_time + TimeDelta(self.nbrUpdates * dt, format="sec"))
        self._update_existing_edges(dt)
        self._add_candidate_edges()

        return (self.graph, self.euro_graph)

    # ──────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────

    @staticmethod
    def _node_id(sat) -> str:
        return f'{sat.plane}-{sat.index}'

    def _init_nodes(self, time):
        euro_nodes = set()
        for sat in self.satellites:
            nid = self._node_id(sat)
            self._id_to_sat[nid] = sat
            self.graph.add_node(nid, position=sat.position)

        if self.only_europe:
            positions = np.stack([sat.position.to(unit.km).value for sat in self.satellites]) * unit.km
            lat, lon = eci_to_latlon_batch(positions, time)
            self._euro_nodes = {
                self._node_id(sat)
                for sat, lat_i, lon_i in zip(self.satellites, lat, lon)
                if self.is_in_europe(lat_i, lon_i)
            }
            self.euro_graph = self.graph.subgraph(self._euro_nodes).copy()

    def _init_edges(self):
        for sat in self.satellites:
            u = self._node_id(sat)
            for neighbor, link_type in self._topology_neighbors(sat):
                v = self._node_id(neighbor)
                if self.only_europe and (u not in self._euro_nodes or v not in self._euro_nodes):
                    continue
                if self.graph.has_edge(u, v):
                    continue
                dist = self._dist_km(sat, neighbor)
                if dist <= self.max_distance_km:
                    self._add_edge(u, v, dist, link_type, state='UP')

    # ──────────────────────────────────────────
    # Update helpers
    # ──────────────────────────────────────────

    def _refresh_positions(self, time):
        for sat in self.satellites:
            # Strip units on every refresh
            self.graph.nodes[self._node_id(sat)]['position'] = sat.position

        if self.only_europe:
            positions = np.stack([sat.position.to(unit.km).value for sat in self.satellites]) * unit.km
            lat, lon = eci_to_latlon_batch(positions, time)
            self._euro_nodes = {
                self._node_id(sat)
                for sat, lat_i, lon_i in zip(self.satellites, lat, lon)
                if self.is_in_europe(lat_i, lon_i)
            }
            self.euro_graph = self.graph.subgraph(self._euro_nodes).copy()

    def _update_existing_edges(self, dt: float):
        to_remove = []

        for u, v, data in self.graph.edges(data=True):
            if self.only_europe and (u not in self._euro_nodes or v not in self._euro_nodes):
                to_remove.append((u, v))
                continue
            if data['link_type'] == LinkType.INTER_PLANE_ISL:
                dist = self._dist_km(self._id_to_sat[u], self._id_to_sat[v])
                if dist > self.max_distance_km:
                    to_remove.append((u, v))
                    continue
            else:
                dist = data['distance']

            data['distance'] = dist
            params           = LINK_PARAMS[data['link_type']]
            if ALWAYS_UP:
                data['state'] = 'UP'
            else:
                data['state']    = _markov_step(data['state'], params, dist, dt)   # 'UP' or 'DOWN'
            data['per']      = _sample_per(params, dist)
            data['latency']  = dist / (C_VACUUM / 1000)

        self.graph.remove_edges_from(to_remove)

    def _add_candidate_edges(self):
        for sat in self.satellites:
            u = self._node_id(sat)
            for neighbor, link_type in self._topology_neighbors(sat):
                v = self._node_id(neighbor)
                if self.only_europe and (u not in self._euro_nodes or v not in self._euro_nodes):
                    continue
                if self.graph.has_edge(u, v):
                    continue
                dist = self._dist_km(sat, neighbor)
                if dist <= self.max_distance_km: # This assumes new links will always be up (irrespective of MTTF). Mostly problematic for European subgraph where links can come in and out of range frequently
                    self._add_edge(u, v, dist, link_type, state='UP') 

    # ──────────────────────────────────────────
    # Topology & geometry
    # ──────────────────────────────────────────

    def _topology_neighbors(self, sat) -> list[tuple]:
        p, s = sat.plane, sat.index

        intra_neighbor = self._plane_to_sat[(p, (s + 1) % self.S)]

        next_plane = (p + 1) % self.P
        inter_neighbors = [
            self._plane_to_sat[(next_plane, t)]
            for t in range(self.S)
        ]
        inter_neighbor = min(
            inter_neighbors,
            key=lambda neighbor: self._dist_km(sat, neighbor)
        )

        return [
            (intra_neighbor, LinkType.INTRA_PLANE_ISL),
            (inter_neighbor, LinkType.INTER_PLANE_ISL),
        ]

    def _add_edge(self, u: str, v: str, distance_km,
                  link_type: LinkType, state: str = 'UP'):
        params = LINK_PARAMS[link_type]
        self.graph.add_edge(u, v,
            link_type = link_type,
            distance  = distance_km,
            state     = state,
            per       = _sample_per(params, distance_km),
            latency   = distance_km / (C_VACUUM / 1000),  # seconds
        )
        if self.only_europe and u in self._euro_nodes and v in self._euro_nodes:
            self.euro_graph.add_edge(u, v,
                link_type = link_type,
                distance  = distance_km,
                state     = state,
                per       = _sample_per(params, distance_km),
                latency   = distance_km / (C_VACUUM / 1000),  # seconds
            )

    @staticmethod
    def is_in_europe(lat, lon):
        # Rough bounding box for Europe
        return 35 <= lat <= 70 and -10 <= lon <= 40

    @staticmethod
    def _dist_km(a, b):
        return float(np.linalg.norm(a.position / unit.km - b.position / unit.km))   # plain float ✓