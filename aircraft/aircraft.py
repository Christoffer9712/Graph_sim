import numpy as np
import networkx as nx


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
        speed_kph         : cruise speed in km/h relative to the ground
        node_id           : graph node identifier string
        """
        self.startPos = np.array(startPos, dtype=float)
        self.destPos  = np.array(destPos,  dtype=float)
        self.node_id  = node_id
        self.position = self.startPos.copy()
        self.arrived  = False
        
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