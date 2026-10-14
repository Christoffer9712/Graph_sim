"""
Nodes, positions and links of the network graph.

Reads the satellite/aircraft positions set by the Propagator and decides
which links exist. Never touches queues or routing.
"""
from utils.network_link_characteristics import markov_step
from utils.utils import haversine_m
from config import (
    FEEDER_LINK_ELEVASION_THRESHOLD_DEG, MAX_ISL_LINK_DISTANCE, SATELLITE_ALTITUDE, LINK_PARAMS, MAX_SAT_PER_GW, MAX_SAT_PER_AC,
    MAX_A2G_LINK_DISTANCE, LinkType, ALWAYS_UP
)
from net_simulation.geometry import dist_m, slant_range_m, is_in_europe


class TopologyModel:
    def __init__(self, graph, satellites, aircrafts, ground_network, P: int, S: int):
        self.graph = graph
        self.satellites = satellites
        self.aircrafts = aircrafts
        self.ground_network = ground_network
        self.P = P
        self.S = S
        self.aircraft_node_ids = {ac.node_id for ac in aircrafts}

        self._plane_to_sat: dict[tuple, object] = {
            (s.plane, s.index): s for s in satellites
        }

        self.max_feeder_link_dist = slant_range_m(SATELLITE_ALTITUDE, FEEDER_LINK_ELEVASION_THRESHOLD_DEG)
        self.max_sa2a_link_dist = self.max_feeder_link_dist  # Assume same for now
        self.max_da2g_link_dist = MAX_A2G_LINK_DISTANCE

        self._topology_neighbors_cache = {}  # Cache topology neighbors for each satellite to avoid repeated calculations
        self.euro_nodes: set[str] = set()

        self._cache_ground_node_lists()  # Pre-compute filtered lists of ground nodes for efficiency
        self._init_nodes()
        self._add_candidate_edges(rebuild_isl=True)

    # ----Public----
    def update(self, dt):
        """Refresh node positions and links for one simulation step of dt seconds."""
        self._refresh_positions()
        self._update_existing_edges(dt)
        self._add_candidate_edges(rebuild_isl=False)

    # ---Initialisation---
    def _cache_ground_node_lists(self):
        """Pre-compute filtered lists of ground nodes to avoid repeated scans."""
        self.gw_nodes = [
            (n, d) for n, d in self.ground_network.nodes(data=True)
            if d['node_type'] == 'gateway'
        ]
        self.av_nodes = [
            (n, d) for n, d in self.ground_network.nodes(data=True)
            if d['node_type'] == 'aviation'
        ]

    def _init_nodes(self):
        for sat in self.satellites:
            self.graph.add_node(self._node_id(sat), node_type='satellite',
                                position=sat.position, velocity=sat.velocity)

        for ac in self.aircrafts:
            self.graph.add_node(ac.node_id,
                                node_type='aircraft',
                                position=ac.position,
                                velocity=ac.velocity)

        self._refresh_euro_nodes()

    # ── Edge builders (shared by init and candidate passes) ──────────────────
    def _build_isl_edges(self, recalculate_neighbors: bool = True) -> set[str]:
        """Add intra- and inter-plane ISL edges. Returns set of European sat IDs."""
        sat_euro_ids: set[str] = set()
        for sat in self.satellites:
            u = self._node_id(sat)
            if recalculate_neighbors:
                neighbors = self._topology_neighbors(sat)
                self._topology_neighbors_cache[u] = neighbors
            if u in self.euro_nodes:
                sat_euro_ids.add(u)
            for neighbor, link_type in self._topology_neighbors_cache[u]:
                v = self._node_id(neighbor)
                if (u not in self.euro_nodes or v not in self.euro_nodes) or self.graph.has_edge(u, v):
                    continue
                dist = dist_m(sat.position, neighbor.position)
                if dist <= MAX_ISL_LINK_DISTANCE:
                    self._add_edge(u, v, dist, link_type, state='UP')
                    self._add_edge(v, u, dist, link_type, state='UP')

        return sat_euro_ids

    def _build_feeder_edges(self, sat_euro_ids: set[str]):
        """Connect gateway nodes to nearby European satellites."""
        for gw_id, gw_data in self.gw_nodes:
            connected = {
                s for s, _, d in self.graph.in_edges(gw_id, data=True)
                if d['link_type'] == LinkType.FEEDER_LINK
            }

            budget = max(0, MAX_SAT_PER_GW - len(connected))

            reachable = self._closest_sats(sat_euro_ids - connected, gw_data['position'], self.max_feeder_link_dist, budget)
            for sat_id, dist in reachable:
                self._add_edge(sat_id, gw_id, dist, LinkType.FEEDER_LINK, state='UP')

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
            for av_id, av_data in self.av_nodes:
                dist = haversine_m(ac.position, av_data['position'])
                if dist <= self.max_da2g_link_dist:
                    self._add_edge(ac.node_id, av_id, dist, LinkType.DA2G, state='UP')

    # ---- Update helpers ----
    def _refresh_positions(self):
        """Copy the positions set by the Propagator into the graph."""
        for sat in self.satellites:
            node = self.graph.nodes[self._node_id(sat)]
            node['position'] = sat.position
            node['velocity'] = sat.velocity

        for ac in self.aircrafts:
            self.graph.nodes[ac.node_id]['position'] = ac.position
            self.graph.nodes[ac.node_id]['velocity'] = ac.velocity

        self._refresh_euro_nodes()

    def _refresh_euro_nodes(self):
        self.euro_nodes = (
            {
                self._node_id(sat)
                for sat in self.satellites
                if is_in_europe(sat.position[0], sat.position[1])
            }
            | set(self.ground_network.nodes())
            | self.aircraft_node_ids
        )

    # ---- Edge Updates ----
    def _update_existing_edges(self, dt: float):
        to_remove = []

        for u, v, data in self.graph.edges(data=True):
            if (u not in self.euro_nodes or v not in self.euro_nodes):
                to_remove.append((u, v))
                continue

            linkType = data['link_type']
            match linkType:
                case LinkType.INTER_PLANE_ISL | LinkType.INTRA_PLANE_ISL:
                    dist = dist_m(self.graph.nodes[u]["position"], self.graph.nodes[v]["position"])
                    if dist > MAX_ISL_LINK_DISTANCE:
                        to_remove.append((u, v))
                        continue

                case LinkType.FEEDER_LINK:
                    if self.graph.nodes[u].get('node_type') == 'gateway':
                        gw_data = self.graph.nodes[u]
                        sat_id = v
                    else:
                        gw_data = self.graph.nodes[v]
                        sat_id = u
                    dist = dist_m(self.graph.nodes[sat_id]["position"], gw_data['position'])
                    if dist > self.max_feeder_link_dist:
                        to_remove.append((u, v))
                        continue

                case LinkType.GROUND_GRID:
                    continue  # Ground links are static

                case LinkType.SA2A:
                    dist = dist_m(self.graph.nodes[u]["position"], self.graph.nodes[v]["position"])
                    if dist > self.max_sa2a_link_dist:
                        to_remove.append((u, v))
                        continue

                case LinkType.DA2G:
                    dist = haversine_m(self.graph.nodes[u]['position'], self.graph.nodes[v]['position'])
                    if dist > self.max_da2g_link_dist:
                        to_remove.append((u, v))
                        continue

                case _:
                    print(f"Unknown link type {linkType} for edge {u} ↔ {v}")
                    continue

            data['distance'] = dist
            data['state'] = 'UP' if ALWAYS_UP else markov_step(data['state'], LINK_PARAMS[linkType], dist, dt)   # 'UP' or 'DOWN'

        self.graph.remove_edges_from(to_remove)

    # ---- Candidate edge addition ----
    def _add_candidate_edges(self, rebuild_isl: bool = False):
        sat_euro_ids = self._build_isl_edges(rebuild_isl)  # Reuse topology neighbors but don't add edges again
        self._build_feeder_edges(sat_euro_ids)
        self._build_a2s_edges(sat_euro_ids)
        self._build_a2g_edges()

    # ---Topology & geometry---
    def _topology_neighbors(self, sat) -> list[tuple]:
        p, s = sat.plane, sat.index

        intra_neighbor = self._plane_to_sat[(p, (s + 1) % self.S)]

        next_plane = (p + 1) % self.P

        inter_neighbor = min(
            (self._plane_to_sat[(next_plane, t)] for t in range(self.S)),
            key=lambda neighbor: dist_m(sat.position, neighbor.position)
        )

        return [
            (intra_neighbor, LinkType.INTRA_PLANE_ISL),
            (inter_neighbor, LinkType.INTER_PLANE_ISL),
        ]

    def _closest_sats(self, sat_ids: set[str], ref_pos, max_range_m: float, limit: int) -> list[tuple]:
        """Return up to `limit` (sat_id, distance_m) pairs within range, sorted closest-first."""
        reachable = []
        for sat_id in sat_ids:
            dist = dist_m(self.graph.nodes[sat_id]['position'], ref_pos)
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

    @staticmethod
    def _node_id(sat) -> str:
        return f'{sat.plane}_{sat.index}'