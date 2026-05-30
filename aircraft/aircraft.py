import numpy as np
import networkx as nx

from config import C_FIBER, C_VACUUM, LINK_PARAMS, LinkType
from .types import TrafficDescription, TunnelDescription
from .base_line_tunnel import tunnel_setup
from utils.network_link_characteristics import sample_per

# 1 degree of arc ≈ 111 km — used to convert speed (km/h) to deg/s.
# Longitude degrees are shorter at high latitudes, but this approximation
# is acceptable for a European simulation where cos(lat) ≈ 0.65–0.85.
_KM_PER_DEG = 111.0 # 6381*2*pi/360

# Per-hop processing baseline added to latency on every link.
_PROCESSING_DELAY_S = 0.0005  # 0.5 ms  — TODO: replace with link-specific model

# Propagation speed by medium.
_SPEED_BY_LINK_TYPE: dict[LinkType, float] = {
    LinkType.SA2A:           C_VACUUM,
    LinkType.DA2G:           C_VACUUM,
    LinkType.INTRA_PLANE_ISL: C_VACUUM,
    LinkType.INTER_PLANE_ISL: C_VACUUM,
    LinkType.FEEDER_LINK:    C_VACUUM,
    LinkType.GROUND_GRID:    C_FIBER,
}

# Bandwidth capacity by link type.
_BW_CAPACITY_BY_LINK_TYPE: dict[LinkType, float] = {
    LinkType.SA2A:              1e9,  # 1 Gbps
    LinkType.DA2G:              1e9,  # 1 Gbps
    LinkType.INTRA_PLANE_ISL:   10e9, # 10 Gbps
    LinkType.INTER_PLANE_ISL:   10e9,  # 10 Gbps
    LinkType.FEEDER_LINK:       1e9,  # 1 Gbps
    LinkType.GROUND_GRID:       10e9, # 10 Gbps
}

# ── Link-level metric models ─────────────────────────────────────────────────
 
def _link_per(link_type: LinkType, total_load_bps: float, distance_m: float) -> float:
    """
    Return PER for a single link given its type and total aggregated load.
 
    Replace the stub below with a proper link-budget / queuing model.
    The load argument is intentionally included in the signature so callers
    do not need to change when the model is upgraded.
 
    Parameters
    ----------
    link_type      : type of the link
    total_load_bps : sum of all flow bandwidths routed over this link (bps)
    distance_m     : distance between the nodes (m)
    """
    if link_type == LinkType.GROUND_GRID:
        per = LINK_PARAMS[link_type].per_ref  # Assume fixed PER for ground links
        utilization = 0.0
    else:
        utilization = total_load_bps / _BW_CAPACITY_BY_LINK_TYPE.get(link_type, 1e9)
        per = sample_per(LINK_PARAMS[link_type], distance_m, utilization)
    #print(f"Computed PER={per:.2e} for link type {link_type} with distance {distance_m:.1f}m and load {total_load_bps:.1f}bps (utilization {utilization:.2%})")
    return per
 
 
def _link_queuing_delay(link_type: LinkType, capacity_bps: float, total_load_bps: float) -> float:
    """
    Return queuing delay for a single link (seconds).
 
    Uses an M/D/1 approximation:  W = ρ / (2μ(1−ρ))
    where ρ = load / capacity  and  μ = capacity (deterministic service rate).
 
    Falls back to zero if capacity is unknown or load exceeds it
    (congestion should be avoided at the routing layer).
 
    Parameters
    ----------
    link_type      : type of the link (reserved for type-specific models)
    capacity_bps   : link capacity in bps (0 means unknown)
    total_load_bps : sum of all flow bandwidths routed over this link (bps)
    """
    if capacity_bps <= 0 or total_load_bps <= 0:
        return 0.0
    rho = min(total_load_bps / capacity_bps, 0.9999)  # clamp to avoid div-by-zero
    return rho / (2 * capacity_bps * (1 - rho))
 
 
# ── Path metric computation ───────────────────────────────────────────────────
 
def compute_path_metrics(
    path_dict: dict[int, list[str] | None],
    graph: nx.Graph,
    flow_bw_dict: dict[int, float],
    source_node_id: str = "",
) -> dict[int, tuple[float, float]]:
    """
    Compute end-to-end (PER, latency) for each flow, accounting for the fact
    that multiple flows may share links and therefore affect each other's
    queuing delay and error rate.
 
    Algorithm
    ---------
    1. Aggregate total bandwidth onto each (u, v) link across all active flows.
    2. Compute per-link PER and queuing delay once, using the aggregated load.
    3. Walk each flow's path and sum the per-link contributions.
 
    Parameters
    ----------
    path_dict      : fiveQI → ordered node list (or None if no path was found)
    graph          : current network snapshot
    flow_bw_dict   : fiveQI → bandwidth demand in bps for that flow
    source_node_id : included in error messages only
 
    Returns
    -------
    fiveQI → (end_to_end_per, total_latency_s)
    Flows with no path are returned as (1.0, inf).
    """
    # ── Step 1: aggregate load per link ──────────────────────────────────────
    # Use a canonical edge key (frozenset) so direction doesn't matter.
    link_load: dict[frozenset, float] = {}
    for fiveQI, path in path_dict.items():
        if path is None:
            continue
        bw = flow_bw_dict.get(fiveQI, 0.0)
        for u, v in zip(path[:-1], path[1:]):
            key = frozenset((u, v))
            link_load[key] = link_load.get(key, 0.0) + bw
 
    # ── Step 2: compute per-link metrics under aggregated load ────────────────
    link_metrics: dict[frozenset, tuple[float, float]] = {}  # key → (per, delay_s)
    for key, total_load in link_load.items():
        u, v = tuple(key)
        edge = graph.get_edge_data(u, v)
        if edge is None:
            suffix = f" (source: {source_node_id})" if source_node_id else ""
            raise ValueError(f"Edge ({u}, {v}) not found in graph{suffix}")
 
        link_type = edge["link_type"]       
        speed = _SPEED_BY_LINK_TYPE.get(link_type)
        if speed is None:
            raise ValueError(f"Edge ({u}, {v}) has unrecognised link type '{link_type}'")
 
        capacity_bps  = _BW_CAPACITY_BY_LINK_TYPE[link_type]
        prop_delay    = edge["distance"] / speed
        queuing_delay = _link_queuing_delay(link_type, capacity_bps, total_load)
        per           = _link_per(link_type, total_load, edge["distance"])
 
        link_metrics[key] = (per, _PROCESSING_DELAY_S + prop_delay + queuing_delay)

    # ── Step 3: accumulate per-flow end-to-end metrics ────────────────────────
    results: dict[int, tuple[float, float]] = {}
    for fiveQI, path in path_dict.items():
        if path is None:
            results[fiveQI] = (1.0, float("inf"))
            continue
 
        success_prob = 1.0
        latency = 0.0
        for u, v in zip(path[:-1], path[1:]):
            per, hop_delay = link_metrics[frozenset((u, v))]
            success_prob *= 1.0 - per
            latency += hop_delay
 
        results[fiveQI] = (1.0 - success_prob, latency)

        print(f'path: {path} for flow {fiveQI} on source {source_node_id} has end-to-end PER={results[fiveQI][0]:.2e} and latency={results[fiveQI][1]:.3f}s')
 
    return results

class Aircraft:
    def __init__(self, startPos, destPos, speed_kph: float, node_id: str):
        """
        Parameters
        ----------
        startPos, destPos : (lat, lon) in degrees
        speed_kph         : cruise speed in km/h relative to the ground
        node_id           : graph node identifier string
        """
        self.startPos = np.array(startPos, dtype=float)
        self.destPos  = np.array(destPos,  dtype=float)
        self.node_id  = node_id
        self.position = self.startPos.copy()
        self.arrived  = False
        self.tunnels: list[TunnelDescription] = []
        self.trafficDemand: dict[int, TrafficDescription] = {}

        direction         = (self.destPos - self.startPos)
        direction         = direction / np.linalg.norm(direction)
        speed_deg_per_s   = speed_kph / _KM_PER_DEG / 3600.0
        self.vel          = speed_deg_per_s * direction

        self.graph = nx.Graph()
        self._sync_graph_node()

    # ── Kinematics ─────────────────────────────────────────────────────────────
    def propagate(self, dt: float) -> None:
        """
        Advance position by dt seconds.
        Clamps to destination once arrived so the node stays in the graph.
        """
        if self.arrived:
            return

        candidate = self.position + self.vel * dt

        # Detect overshoot: remaining vector and movement vector point opposite ways
        remaining = self.destPos - self.position
        if (np.dot(remaining, candidate - self.position) < 0 or
                np.linalg.norm(candidate - self.destPos) < 0.01):
            candidate    = self.destPos.copy()
            self.arrived = True

        self.position = candidate
        self._sync_graph_node()

    # ── Traffic and tunnels ───────────────────────────────────────────────────
    def setTrafficDemand(self, trafficDemand: dict[int, TrafficDescription]) -> None:
        self.trafficDemand = trafficDemand

    def setUpTunnels(self, dt_tunnel: float, graph: nx.Graph) -> None:
        links = list(graph.neighbors(self.node_id))
        self.tunnels = tunnel_setup(self.node_id, dt_tunnel, graph, self.trafficDemand)
        #print(f"Aircraft {self.node_id} has links: {links}")
        #print(f"Aircraft {self.node_id} set up tunnels: {self.tunnels}")
    
    # ── Data plane ────────────────────────────────────────────────────────────
    def sendData(
        self,
        traffic: dict[int, TrafficDescription],
        graph:   nx.Graph,
    ) -> dict[int, tuple[float, float]]:
        metrics_dict = {} # fiveQI -> (PER, latency). Works because we aggregate traffic by 5QI. Also, assume at most one tunnel per 5QI
        path_dict = {} # fiveQI -> path list
        flow_bw_dict = {} # fiveQI -> bandwidth demand in bps
        for fiveQI, desc in traffic.items():
            tunnel = self._mapToTunnel(desc)
            if tunnel is None:
                print(f"No tunnel found for traffic demand {desc} on aircraft {self.node_id}")
                path_dict[fiveQI] = None
                continue
            
            path = self._getPath(tunnel, graph)
            if path is None:
                print(f"Path not found for tunnel {tunnel} on aircraft {self.node_id}")
            
            path_dict[desc.fiveQI] = path
            flow_bw_dict[fiveQI] = desc.BW

        metrics_dict = compute_path_metrics(path_dict, graph, flow_bw_dict, self.node_id)

        return metrics_dict

    # ── Private Helpers ──────────────────────────────────────────────────────────────
    def _sync_graph_node(self) -> None:
        """Write current position into the graph node attributes."""
        lat, lon = float(self.position[0]), float(self.position[1])
        if self.node_id in self.graph:
            self.graph.nodes[self.node_id].update(
                node_type='aircraft',
                position=(lat, lon),
                vel = (self.vel[0], self.vel[1]) # degrees/s in lat/lon directions
            )
        else:
            self.graph.add_node(
                self.node_id,
                node_type='aircraft',
                position=(lat, lon),
                vel = (self.vel[0], self.vel[1]) # degrees/s in lat/lon directions
            )
    
    def _mapToTunnel(self, desc: TrafficDescription) -> TunnelDescription | None:
        """
        Return the first established tunnel that satisfies the QoS class and BW.
        A production version would track remaining capacity per tunnel and
        subtract reserved BW after each mapping.
        """
        for tunnel in self.tunnels:
            if tunnel.fiveQI == desc.fiveQI and tunnel.BW >= desc.BW:
                return tunnel
        return None

    def _getPath(
        self, tunnel: TunnelDescription, graph: nx.Graph
    ) -> list | None:
        """
        Construct the ordered node list for the tunnel end-to-end.
        The [1:] slices remove the duplicate junction node at each segment join.
        Returns None if any segment is unreachable in the current snapshot.
        """
        try:
            path = nx.shortest_path(graph, self.node_id, tunnel.firstHop)
            if len(path) > 2:
                return None  # Unreachable first hop (should be directly connected)
            if tunnel.linkType == LinkType.DA2G:
                path += nx.shortest_path(graph, tunnel.firstHop, tunnel.UPF)[1:]

            elif tunnel.linkType == LinkType.SA2A:
                path += nx.shortest_path(graph, tunnel.firstHop, tunnel.GW)[1:]
                path += nx.shortest_path(graph, tunnel.GW,       tunnel.UPF)[1:]

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        return path