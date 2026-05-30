from unittest import case

import networkx as nx
import numpy as np
import astropy.units as unit

from astropy.time import TimeDelta, Time
from astropy.coordinates import EarthLocation

from orbit.satellite import Satellite
from aircraft.aircraft import Aircraft
from ground_network.nodes import GroundNodeType
from utils.coordinates import eci_to_latlon_batch
from utils.network_link_characteristics import markov_step
from utils.utils import haversine_m
from config import (
    FEEDER_LINK_ELEVASION_THRESHOLD_DEG, MAX_ISL_LINK_DISTANCE, SATELLITE_ALTITUDE, LINK_PARAMS, MAX_SAT_PER_GW, MAX_SAT_PER_AC,
    MAX_A2G_LINK_DISTANCE, LinkType, ALWAYS_UP, AIRCRRAFT_ALTITUDE
)

class DynamicNetwork:
    def __init__(self, satellites: list[Satellite], P: int, S: int, F: int, start_time: Time, ground_network: nx.Graph, aircrafts: list[Aircraft], only_europe: bool = False):
        self.satellites = satellites
        self.P = P
        self.S = S
        self.F = F
        self.only_europe = only_europe
        self.start_time = start_time
        self.ground_network = ground_network
        self.aircrafts = aircrafts
        self.aircraft_node_ids = {ac.node_id for ac in aircrafts}

        self._id_to_sat: dict[str, object] = {}
        self._plane_to_sat: dict[tuple, object] = {
            (s.plane, s.index): s for s in satellites
        }

        self.max_feeder_link_dist = self.slant_range_m(float(SATELLITE_ALTITUDE.to_value(unit.m)), FEEDER_LINK_ELEVASION_THRESHOLD_DEG)
        self.max_sa2a_link_dist = self.max_feeder_link_dist  # Assume same for now
        self.max_da2g_link_dist = MAX_A2G_LINK_DISTANCE

        self._topology_neighbors_cache = {}  # Cache topology neighbors for each satellite to avoid repeated calculations

        self.graph = ground_network.copy()  # Start with the ground network; we'll add satellites and ISLs to this
        self.euro_graph = nx.Graph()  # Subgraph view containing only European satellites and all ground nodes
        self._euro_nodes: set[str] = set()
        self.nbrUpdates = 0

        self._cache_ground_node_lists()  # Pre-compute filtered lists of ground nodes for efficiency
        self._init_nodes(self.start_time)
        self._add_candidate_edges(self.start_time, rebuild_isl=True)
    
    # ----Public----
    def update(self, dt) -> tuple[nx.Graph, nx.Graph]:
        """
        Incremental update for one simulation step.

        Parameters
        ──────────
        dt : step duration in seconds
        """
        self.nbrUpdates += 1
        t = self.start_time + TimeDelta(self.nbrUpdates * dt, format="sec")
        self._refresh_positions(t)
        self._update_existing_edges(dt,t)
        self._add_candidate_edges(t, rebuild_isl=False)
        
        self.euro_graph = self.graph.subgraph(self._euro_nodes).copy()
        return (self.graph, self.euro_graph)

    # ---Initialisation---
    def _cache_ground_node_lists(self):
        """Pre-compute filtered lists of ground nodes to avoid repeated scans."""
        self.gw_nodes = [
            (n, d) for n, d in self.ground_network.nodes(data=True)
            if d['node_type'] == GroundNodeType.GATEWAY
        ]
        self.av_nodes = [
            (n, d) for n, d in self.ground_network.nodes(data=True)
            if d['node_type'] == GroundNodeType.AVIATION
        ]

    def _init_nodes(self, time):
        for sat in self.satellites:
            nid = self._node_id(sat)
            self._id_to_sat[nid] = sat
            self.graph.add_node(nid, node_type='satellite', position=sat.position)

        for ac in self.aircrafts:
            lat, lon = ac.position
            eci_pos = self.gs_to_eci(lat, lon, AIRCRRAFT_ALTITUDE, time)
            self.graph.add_node(ac.node_id,
                                node_type='aircraft',
                                position=eci_pos,
                                lat=ac.position[0],
                                lon=ac.position[1],
                                vel=ac.vel)

        self._refresh_euro_nodes(time)

    # ── Edge builders (shared by init and candidate passes) ──────────────────
    def _build_isl_edges(self, recalculate_neighbors: bool = True) -> set[str]:
        """Add intra- and inter-plane ISL edges. Returns set of European sat IDs."""
        sat_euro_ids: set[str] = set()
        for sat in self.satellites:
            u = self._node_id(sat)
            if recalculate_neighbors:
                neighbors = self._topology_neighbors(sat)
                self._topology_neighbors_cache[u] = neighbors
            if u in self._euro_nodes:
                sat_euro_ids.add(u)
            for neighbor, link_type in self._topology_neighbors_cache[u]:
                v = self._node_id(neighbor)
                if self.only_europe and (u not in self._euro_nodes or v not in self._euro_nodes):
                    continue
                if self.graph.has_edge(u, v):
                    continue
                dist = self._dist_m(sat.position, neighbor.position)
                if dist <= MAX_ISL_LINK_DISTANCE:
                    self._add_edge(u, v, dist, link_type, state='UP')

        return sat_euro_ids

    def _build_feeder_edges(self, sat_euro_ids: set[str], time):
        """
        Connect gateway nodes to nearby European satellites.
        """
        
        for gw_id, gw_data in self.gw_nodes:
            gw_pos = self.gs_to_eci(gw_data['lat'], gw_data['lon'], 0, time)

            existing = (
                sum(
                    1 for _, _, d in self.graph.edges(gw_id, data=True)
                    if d['link_type'] == LinkType.FEEDER_LINK
                )
            )

            budget = (MAX_SAT_PER_GW - existing)

            reachable = self._closest_sats(sat_euro_ids, gw_pos, self.max_feeder_link_dist, budget)
            for sat_id, dist in reachable:
                self._add_edge(gw_id, sat_id, dist, LinkType.FEEDER_LINK, state='UP')
        
    def _build_a2s_edges(self, sat_euro_ids: set[str]):
        """Connect each aircraft to its nearest European satellites."""
        for ac in self.aircrafts:
            ac_pos = self.graph.nodes[ac.node_id]['position']
            reachable = self._closest_sats(sat_euro_ids, ac_pos, self.max_sa2a_link_dist, MAX_SAT_PER_AC)

            # Remove stale links that are no longer among the best candidates
            best_ids = {sat_id for sat_id, _ in reachable}
            stale = [
                (ac.node_id, nbr)
                for nbr in self.graph.neighbors(ac.node_id)
                if (
                    self.graph.edges[ac.node_id, nbr]['link_type'] == LinkType.SA2A
                    and nbr not in best_ids
                )
            ]
            self.graph.remove_edges_from(stale)

            for sat_id, dist in reachable:
                if not self.graph.has_edge(ac.node_id, sat_id):
                    self._add_edge(ac.node_id, sat_id, dist, LinkType.SA2A, state='UP')

    def _build_a2g_edges(self):
        """Connect each aircraft to all aviation ground nodes within range."""
        for ac in self.aircrafts:
            ac_lat, ac_lon = ac.position
            for av_id, av_data in self.av_nodes:
                dist_m = haversine_m((ac_lat, ac_lon), (av_data['lat'], av_data['lon']))
                if dist_m <= self.max_da2g_link_dist:
                    self._add_edge(ac.node_id, av_id, dist_m, LinkType.DA2G, state='UP')

    
    # ---- Update helpers ----
    def _refresh_positions(self, time):
        for sat in self.satellites:
            self.graph.nodes[self._node_id(sat)]['position'] = sat.position

        for ac in self.aircrafts:
            lat, lon = ac.position
            eci_pos = self.gs_to_eci(lat, lon, AIRCRRAFT_ALTITUDE, time)
            self.graph.nodes[ac.node_id]['position'] = eci_pos
            self.graph.nodes[ac.node_id]['lat'] = lat
            self.graph.nodes[ac.node_id]['lon'] = lon

        if self.only_europe:
            self._refresh_euro_nodes(time)

    def _refresh_euro_nodes(self, time):
        positions = np.stack([sat.position.to(unit.m).value for sat in self.satellites]) * unit.m
        lat, lon = eci_to_latlon_batch(positions, time)
        self._euro_nodes = (
            {
                self._node_id(sat)
                for sat, lat_i, lon_i in zip(self.satellites, lat, lon)
                if self.is_in_europe(lat_i, lon_i)
            }
            | set(self.ground_network.nodes())
            | self.aircraft_node_ids
        )
        self.euro_graph = self.graph.subgraph(self._euro_nodes).copy()
    
    # ---- Edge Updates ----
    def _update_existing_edges(self, dt: float, time):
        to_remove = []

        for u, v, data in self.graph.edges(data=True):
            if self.only_europe and (u not in self._euro_nodes or v not in self._euro_nodes):
                to_remove.append((u, v))
                continue
            
            linkType = data['link_type']
            match linkType:
                case LinkType.INTER_PLANE_ISL | LinkType.INTRA_PLANE_ISL:
                    dist_m = self._dist_m(self.graph.nodes[u]["position"], self.graph.nodes[v]["position"])
                    if dist_m > MAX_ISL_LINK_DISTANCE:
                        to_remove.append((u, v))
                        continue

                case LinkType.FEEDER_LINK:
                    if self.graph.nodes[u].get('node_type') == GroundNodeType.GATEWAY:
                        gw_data = self.graph.nodes[u]
                        sat_id = v
                    else:
                        gw_data = self.graph.nodes[v]
                        sat_id = u
                    gw_pos = self.gs_to_eci(gw_data['lat'], gw_data['lon'], 0, time)
                    dist_m = self._dist_m(self.graph.nodes[sat_id]["position"], gw_pos)
                    if dist_m > self.max_feeder_link_dist:
                        to_remove.append((u, v))
                        continue
            
                case LinkType.GROUND_GRID:
                    continue  # Ground links are static

                case LinkType.SA2A:
                    dist_m = self._dist_m(self.graph.nodes[u]["position"], self.graph.nodes[v]["position"])
                    if dist_m > self.max_sa2a_link_dist:
                        to_remove.append((u, v))
                        continue
                
                case LinkType.DA2G:
                    dist_m = haversine_m((self.graph.nodes[u]['lat'], self.graph.nodes[u]['lon']), (self.graph.nodes[v]['lat'], self.graph.nodes[v]['lon']))
                    if dist_m > self.max_da2g_link_dist:
                        to_remove.append((u, v))
                        continue

                case _:
                    print(f"Unknown link type {linkType} for edge {u} ↔ {v}")
                

            data['distance'] = dist_m
            data['state'] = 'UP' if ALWAYS_UP else markov_step(data['state'], LINK_PARAMS[linkType], dist_m, dt)   # 'UP' or 'DOWN'

        self.graph.remove_edges_from(to_remove)

# ---- Candidate edge addition ----
    def _add_candidate_edges(self, time, rebuild_isl: bool = False):
        sat_euro_ids = self._build_isl_edges(rebuild_isl) # Reuse topology neighbors but don't add edges again
        self._build_feeder_edges(sat_euro_ids, time)
        self._build_a2s_edges(sat_euro_ids)
        self._build_a2g_edges()

    # ---Topology & geometry---

    def _topology_neighbors(self, sat) -> list[tuple]:
        p, s = sat.plane, sat.index

        intra_neighbor = self._plane_to_sat[(p, (s + 1) % self.S)]

        next_plane = (p + 1) % self.P

        inter_neighbor = min(
            (self._plane_to_sat[(next_plane, t)] for t in range(self.S)),
            key=lambda neighbor: self._dist_m(sat.position, neighbor.position)
        )

        return [
            (intra_neighbor, LinkType.INTRA_PLANE_ISL),
            (inter_neighbor, LinkType.INTER_PLANE_ISL),
        ]

    def _closest_sats(self, sat_ids: set[str], ref_pos, max_range_m: float, limit: int) -> list[tuple]:
        """Return up to `limit` (sat_id, distance_m) pairs within range, sorted closest-first."""
        reachable = []
        for sat_id in sat_ids:
            dist = self._dist_m(self.graph.nodes[sat_id]['position'], ref_pos)
            if dist <= max_range_m:
                reachable.append((sat_id, dist))
        reachable.sort(key=lambda x: x[1])
        return reachable[:limit]

    def _add_edge(self, u: str, v: str, distance_m, link_type: LinkType, state: str = 'UP'):
        self.graph.add_edge(u, v,
            link_type = link_type,
            distance  = distance_m,
            state     = state,
        )
        if self.only_europe and u in self._euro_nodes and v in self._euro_nodes:
            self.euro_graph.add_edge(u, v,
                link_type = link_type,
                distance  = distance_m,
                state     = state,
            )

    # ─── Static helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def gs_to_eci(lat, lon, alt, t):
        loc = EarthLocation(
            lat=lat * unit.deg,
            lon=lon * unit.deg,
            height=alt * unit.m
        )

        gcrs = loc.get_gcrs(obstime=t)
        return gcrs.cartesian.xyz.to(unit.km)

    @staticmethod
    def slant_range_m(sat_alt_m: float, elevation_deg: float) -> float:
        """
        Approximate slant range from ground to satellite given altitude and elevation angle.
        Derived from the law of sines on the Earth-centre / ground-station / satellite triangle.
        The flat-Earth approximation sat_alt / sin(el) is used for el > 10° and is accurate
        to within ~2% at 20° elevation for LEO altitudes.
        """
        el_rad = np.radians(max(elevation_deg, 1.0))   # clamp avoids division by zero
        return sat_alt_m / np.sin(el_rad)

    @staticmethod
    def is_in_europe(lat, lon):
        # Rough bounding box for Europe
        return 35 <= lat <= 70 and -10 <= lon <= 40

    @staticmethod
    def _dist_m(a_pos, b_pos) -> float:
        return float(np.linalg.norm((a_pos - b_pos).to(unit.m).value))
    
    @staticmethod
    def _node_id(sat) -> str:
        return f'{sat.plane}-{sat.index}'