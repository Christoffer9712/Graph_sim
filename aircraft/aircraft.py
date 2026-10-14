import math
import networkx as nx
from config import AIRCRRAFT_ALTITUDE

# 1 degree of arc ≈ 111 km — used to convert speed (km/h) to deg/s.
# Longitude degrees are shorter at high latitudes, but this approximation
# is acceptable for a European simulation where cos(lat) ≈ 0.65–0.85.
_KM_PER_DEG = 111.0 # 6381*2*pi/360

class Aircraft:
    def __init__(self, startPos, destPos, speed_kph: float, node_id: str, altitude=AIRCRRAFT_ALTITUDE):
        """
        Parameters
        ----------
        startPos, destPos : (lat, lon) in degrees
        speed_kph         : cruise speed in km/h relative to the ground
        node_id           : graph node identifier string
        """
        self.startPos = [float(startPos[0]), float(startPos[1])]
        self.destPos  = [float(destPos[0]), float(destPos[1])]
        self.node_id  = node_id
        self.position = self.startPos + [altitude]
        self.arrived  = False

        d_lat             = self.destPos[0] - self.startPos[0]
        d_lon             = self.destPos[1] - self.startPos[1]
        norm              = math.hypot(d_lat, d_lon)
        speed_deg_per_s   = speed_kph / _KM_PER_DEG / 3600.0
        self.velocity     = (speed_deg_per_s * d_lat / norm, speed_deg_per_s * d_lon / norm)
        self.altitude       = altitude
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

        lat, lon = self.position[0], self.position[1]
        cand_lat = lat + self.velocity[0] * dt
        cand_lon = lon + self.velocity[1] * dt

        # Detect overshoot: remaining vector (after the move) and movement vector point opposite ways
        remaining = (self.destPos[0] - cand_lat, self.destPos[1] - cand_lon)
        movement  = (cand_lat - lat, cand_lon - lon)
        if (remaining[0] * movement[0] + remaining[1] * movement[1] < 0 or
                math.hypot(remaining[0], remaining[1]) < 0.01):
            cand_lat, cand_lon = self.destPos
            self.arrived = True

        self.position = [cand_lat, cand_lon, self.altitude]
        self._sync_graph_node()

    # ── Private Helpers ──────────────────────────────────────────────────────────────
    def _sync_graph_node(self) -> None:
        """Write current position into the graph node attributes."""
        lat, lon = float(self.position[0]), float(self.position[1])
        if self.node_id in self.graph:
            self.graph.nodes[self.node_id].update(
                node_type='aircraft',
                position=(lat, lon, self.altitude),
                velocity = (self.velocity[0], self.velocity[1]) # degrees/s in lat/lon directions
            )
        else:
            self.graph.add_node(
                self.node_id,
                node_type='aircraft',
                position=(lat, lon, self.altitude),
                velocity = (self.velocity[0], self.velocity[1]) # degrees/s in lat/lon directions
            )