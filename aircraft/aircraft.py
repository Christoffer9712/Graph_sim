import numpy as np
import networkx as nx
from .types import TrafficDescription, TunnelDescription, LinkType
from astropy import units as u
# 1 degree of arc ≈ 111 km — used to convert speed (km/h) to deg/s.
# Longitude degrees are shorter at high latitudes, but this approximation
# is acceptable for a European simulation where cos(lat) ≈ 0.65–0.85.
_KM_PER_DEG = 111.0 # 6381*2*pi/360


class Aircraft:
    def __init__(self, startPos, destPos, speed_kph: float, node_id: str):
        """
        Parameters
        ----------
        startPos, destPos : (lat, lon) in degrees
        speed_kph         : cruise speed in km/h
        node_id           : graph node identifier string
        """
        self.startPos = np.array(startPos, dtype=float)
        self.destPos  = np.array(destPos,  dtype=float)
        self.node_id  = node_id
        self.position = self.startPos.copy()
        self.arrived  = False

        # Direction unit vector in lat/lon space, velocity in deg/s
        direction         = (self.destPos - self.startPos) / np.linalg.norm(self.destPos - self.startPos)
        speed_deg_per_s   = speed_kph / _KM_PER_DEG / 3600.0
        self.vel          = speed_deg_per_s * direction

        # The aircraft's own single-node graph, merged into the full graph each step.
        # We store both 'position' (tuple) for topology checks and
        # 'lat'/'lon' scalars for convenience elsewhere.
        self.graph = nx.Graph()
        self._sync_graph_node()

        self.trafficDemand: list[TrafficDescription] = []
        self.tunnels:       list[TunnelDescription]  = []

    # ── internal ──────────────────────────────────────────────────────────────

    def _sync_graph_node(self) -> None:
        """Write current position into the graph node attributes."""
        lat, lon = float(self.position[0]), float(self.position[1])
        if self.node_id in self.graph:
            self.graph.nodes[self.node_id].update(
                type='aircraft',
                position=(lat, lon),
                lat=lat,
                lon=lon
            )
        else:
            self.graph.add_node(
                self.node_id,
                node_type='aircraft',
                position=(lat, lon),
                lat=lat,
                lon=lon,
            )

    # ── movement ──────────────────────────────────────────────────────────────

    def propagate(self, dt: float) -> None:
        """
        Advance position by dt seconds along the great-circle approximation.
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

    # ── traffic and tunnels ───────────────────────────────────────────────────

    def setTrafficDemand(self, trafficDemand: list[TrafficDescription]) -> None:
        self.trafficDemand = list(trafficDemand)

    def setUpTunnels(self, dt_tunnel: float, graph: nx.Graph) -> None:
        # Lazy import avoids the circular dependency between aircraft ↔ ml
        
        #from aircraft import ml as ML
        #self.tunnels = ML.get_tunnels(
        #    self.trafficDemand, dt_tunnel, graph, self.node_id
        #)
        for edge in graph.edges([self.node_id]):
            print(f"Aircraft {self.node_id} has edge: {edge}")
        
        links = list(graph.neighbors(self.node_id))

        print(f"Aircraft {self.node_id} has links: {links}")
        self.tunnels = [TunnelDescription(
            fiveQI=desc.fiveQI,
            BW=desc.BW,
            linkType=LinkType.SA2A,  # Placeholder; replace with actual link type
            firstHop=links[0],             # Placeholder; replace with actual first hop
            GW='GW_NL_Burum',                   # Placeholder; replace with actual GW
            UPF=desc.UPF                   # Placeholder; replace with actual UPF
        ) for desc in self.trafficDemand]
        print(f"Aircraft {self.node_id} set up tunnels: {self.tunnels}")
    # ── data plane ────────────────────────────────────────────────────────────

    def sendData(
        self,
        traffic: list[TrafficDescription],
        graph:   nx.Graph,
    ) -> tuple[list[float], list[float]]:
        per_list, latency_list = [], []
        for desc in traffic:
            tunnel = self.mapToTunnel(desc)
            if tunnel is None:
                print(f"No tunnel found for traffic demand {desc} on aircraft {self.node_id}")
                per_list.append(1.0)
                latency_list.append(float('inf'))
                continue
            path = self.getPath(tunnel, graph)
            if path is None:
                print(f"Path not found for tunnel {tunnel} on aircraft {self.node_id}")
                per_list.append(1.0)
                latency_list.append(float('inf'))
                continue
            per, latency = self.getPathMetrics(path, graph)
            per_list.append(per)
            latency_list.append(latency)
            print(f"Aircraft {self.node_id} traffic {desc}: path={path}, PER={per:.2e}, latency={latency:.3f}s")
        return per_list, latency_list

    def mapToTunnel(self, desc: TrafficDescription) -> TunnelDescription | None:
        """
        Return the first established tunnel that satisfies the QoS class and BW.
        A production version would track remaining capacity per tunnel and
        subtract reserved BW after each mapping.
        """
        for tunnel in self.tunnels:
            if tunnel.fiveQI == desc.fiveQI and tunnel.BW >= desc.BW:
                return tunnel
        return None

    def getPath(
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

    def getPathMetrics(
        self, path: list, graph: nx.Graph
    ) -> tuple[float, float]:
        """
        Walk the path hop-by-hop and accumulate:
          PER     — end-to-end packet error rate (multiplicative through link success probs)
          latency — sum of per-hop propagation delays in seconds

        PER_total = 1 − ∏(1 − PER_link)
        Falls back to conservative defaults if edge attributes are missing
        (e.g. ISL edges that predate the latency/per additions).
        """
        success_prob = 1.0
        latency      = 0.0
        for u, v in zip(path[:-1], path[1:]):
            edge = graph.get_edge_data(u, v) or {}
            #success_prob *= 1.0 - edge.get('per',     1e-4)
            #latency      +=       edge.get('latency', 5e-3)   # 5 ms default

            print(f"Edge ({u}, {v}) attributes: {edge}")
            success_prob *= 1.0 - edge['per']
            latency      +=       edge['latency']
        return 1.0 - success_prob, latency