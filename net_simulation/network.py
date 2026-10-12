from unittest import case

import networkx as nx
import numpy as np
import astropy.units as unit
import math

from astropy.time import TimeDelta, Time
from astropy.coordinates import EarthLocation

from orbit.satellite import Satellite
from aircraft.aircraft import Aircraft
from utils.coordinates import eci_to_latlon_batch
from utils.network_link_characteristics import markov_step
from utils.utils import haversine_m
from config import (
    FEEDER_LINK_ELEVASION_THRESHOLD_DEG, MAX_ISL_LINK_DISTANCE, SATELLITE_ALTITUDE, LINK_PARAMS, MAX_SAT_PER_GW, MAX_SAT_PER_AC,
    MAX_A2G_LINK_DISTANCE, LinkType, ALWAYS_UP, AIRCRRAFT_ALTITUDE, C_VACUUM, C_FIBER
)

class DynamicNetwork:
    def __init__(self, satellites: list[Satellite], P: int, S: int, F: int, start_time: Time, ground_network: nx.Graph, aircrafts: list[Aircraft]):
        self.satellites = satellites
        self.P = P
        self.S = S
        self.F = F
        self.start_time = start_time
        self.ground_network = ground_network
        self.aircrafts = aircrafts
        self.aircraft_node_ids = {ac.node_id for ac in aircrafts}

        self._id_to_sat: dict[str, object] = {}
        self._plane_to_sat: dict[tuple, object] = {
            (s.plane, s.index): s for s in satellites
        }

        self.rng = np.random.default_rng(42)

        self.max_feeder_link_dist = self.slant_range_m(SATELLITE_ALTITUDE, FEEDER_LINK_ELEVASION_THRESHOLD_DEG)
        self.max_sa2a_link_dist = self.max_feeder_link_dist  # Assume same for now
        self.max_da2g_link_dist = MAX_A2G_LINK_DISTANCE

        self._topology_neighbors_cache = {}  # Cache topology neighbors for each satellite to avoid repeated calculations

        self.graph = ground_network.to_directed()  # Start with the ground network; we'll add satellites and ISLs to this
        self.euro_graph = nx.DiGraph()  # Subgraph view containing only European satellites and all ground nodes
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
        self._step_queue(dt)
        self.euro_graph = self.graph.subgraph(self._euro_nodes).copy()
        return (self.graph, self.euro_graph)

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
        self._init_ou_params()

    def _init_ou_params(self,
                        service_rate_range=(0.8, 1.2),      # mu, per node type below
                        base_load_range=(0.3, 0.6),    # mean rho each node sits at
                        ou_theta= 0.006,                # mean-reversion rate (1/timescale)
                        ou_sigma=0.01,                  # driving noise magnitude
                        rho_max=0.9,                  # clamp to keep M/M/1 finite
                        rho_min=0.02,                  # lower clamp of the utilisation
                        burn_in_time=None,             # extra warm-up before t=0 (None -> 2 / ou_theta)
                        burn_in_dt=1.0                 # step used during the warm-up (use the simulation dt)
                        ):
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.rho_max = rho_max
        self.rho_min = rho_min
        self.type_mu = {'aircraft':  400,    # access link: ~5 ms (the real bottleneck)
                        'satellite': 1000,
                        'gateway':   1000,
                        'aviation':  1000,
                        'upf':       1000}

        for _, d in self.graph.nodes(data=True):
            mult = self.type_mu.get(d["node_type"])
            mu = self.rng.uniform(*service_rate_range) * mult
            rho0 = self.rng.uniform(*base_load_range)
            #mu = 400 #!!! OBS !!!
            #rho0 = 0.5 #!!! OBS !!!
            d["service_rate"] = mu
            d["base_rho"] = rho0          # OU mean, in utilisation units

            # Stationary std of the (unclipped) O-U utilisation: sigma / sqrt(2 theta)
            stat_std = ou_sigma / math.sqrt(2.0 * self.ou_theta)
            
            # Initial state: a draw from the stationary distribution N(rho0, sigma^2 / (2 theta))
            # instead of the mean itself. Starting at the mean gives zero variance at t=0
            # that only builds up over ~1/(2 theta) steps, so E[queue_delay] (convex in rho)
            # would drift upwards during the simulation.
            rho_init = float(np.clip(rho0 + stat_std * self.rng.standard_normal(), self.rho_min, self.rho_max))
            d["rho_state"] = rho_init     # OU state
            d["utilisation"] = rho_init
            d["load"] = rho_init * mu     # lambda
            d["queue_len"] = self._equilibrium_len(rho_init, rho_max=self.rho_max)
            d["queue_delay"] = self._equilibrium_delay(rho_init, mu, rho_max=self.rho_max)

        # Warm-up: the clamp to [rho_min, rho_max] makes the true stationary distribution
        # slightly non-Gaussian; running the process for a while before t=0 removes the
        # remaining mismatch (it decays like exp(-theta * t)).
        if burn_in_time is None:
            burn_in_time = 2.0 / self.ou_theta
        for _ in range(int(math.ceil(burn_in_time / burn_in_dt))):
            self._step_queue(burn_in_dt)

    def _equilibrium_len(self, rho, rho_max):
        """L = rho / (1 - rho)."""
        rho = min(rho, rho_max)
        return rho / (1.0 - rho)

    def _equilibrium_delay(self, rho, mu, rho_max):
        """W = 1 / (mu - lambda) = 1 / (mu (1 - rho))."""
        rho = min(rho, rho_max)
        return 1.0 / (mu * (1.0 - rho))
    
    def _step_queue(self, dt):
        """
        Advance all node queues by dt.

        """
        for n, d in self.graph.nodes(data=True):
            mu = d["service_rate"]

            # --- 1. Offered load: Ornstein-Uhlenbeck, mean-reverting ---------
            mean = d["base_rho"]
            drift = self.ou_theta * (mean - d["rho_state"]) * dt
            noise = self.ou_sigma * math.sqrt(dt) * self.rng.standard_normal()
            rho = d["rho_state"] + drift + noise
            rho = float(np.clip(rho, self.rho_min, self.rho_max))
            d["rho_state"] = rho
            d["utilisation"] = rho
            d["load"] = rho * mu

            # --- 2. Queue relaxes toward equilibrium, not instantly ----------
            #target_len = self._equilibrium_len(rho)
            #alpha = 1.0 - math.exp(-dt / self.queue_tau)
            #d["queue_len"] += alpha * (target_len - d["queue_len"])
            d["queue_len"] = self._equilibrium_len(rho, self.rho_max)

            # --- 3. Derived metrics ------------------------------------------
            # Delay from actual occupancy via Little's law: W = L / lambda
            d["queue_delay"] = d["queue_len"] / max(d["load"], 1e-6)

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
                if (u not in self._euro_nodes or v not in self._euro_nodes) or self.graph.has_edge(u, v):
                    continue
                dist = self._dist_m(sat.position, neighbor.position)
                if dist <= MAX_ISL_LINK_DISTANCE:
                    self._add_edge(u, v, dist, link_type, state='UP')
                    self._add_edge(v, u, dist, link_type, state='UP')

        return sat_euro_ids

    def _build_feeder_edges(self, sat_euro_ids: set[str], time):
        """
        Connect gateway nodes to nearby European satellites.
        """
        
        for gw_id, gw_data in self.gw_nodes:
            gw_pos = self.gs_to_eci(gw_data['lat'], gw_data['lon'], 0, time)

            connected = {
                s for s, _, d in self.graph.in_edges(gw_id, data=True)
                if d['link_type'] == LinkType.FEEDER_LINK
            }

            budget = max(0, MAX_SAT_PER_GW - len(connected))

            reachable = self._closest_sats(sat_euro_ids-connected, gw_pos, self.max_feeder_link_dist, budget)
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
            if (u not in self._euro_nodes or v not in self._euro_nodes):
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
                    if self.graph.nodes[u].get('node_type') == 'gateway':
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
                    continue

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
        if u in self._euro_nodes and v in self._euro_nodes:
            self.euro_graph.add_edge(u, v,
                link_type = link_type,
                distance  = distance_m,
                state     = state,
            )

    def _get_path_distance_delay(self, path, time, calc_delay = False):
        tot_dist = 0
        prop_delay = 0
        for node1, node2 in zip(path[:-1], path[1:]):
            n1 = self.graph.nodes[node1]
            n2 = self.graph.nodes[node2]
            if 'position' in n1.keys():
                p1 = n1['position']
            else:
                p1 = self.gs_to_eci(n1['lat'], n1['lon'], 0, time)
            if 'position' in n2.keys():
                p2 = n2['position']
            else:
                p2 = self.gs_to_eci(n2['lat'], n2['lon'], 0, time)
            dist = self._dist_m(p1, p2)
            tot_dist += dist
            if calc_delay:
                if self.graph.get_edge_data(node1, node2)['link_type'] == LinkType.GROUND_GRID:
                    prop_delay += dist/C_FIBER
                else:
                    prop_delay += dist/C_VACUUM

        return (tot_dist, prop_delay)
        
    def get_udl_path_delay(self, udl, time):
        queue_delay = 0
        (_, prop_delay) = self._get_path_distance_delay(udl, time, calc_delay=True)
        for node in udl:
            n1 = self.graph.nodes[node]
            queue_delay += n1['queue_delay']

        print(f'prop_delay/tot_delay = {prop_delay/(prop_delay + queue_delay)}')
        return(queue_delay + prop_delay)

    def _delay_weight(self, u, v, d):
        if d.get('state') != 'UP':
            return None
        #c = C_FIBER if d['link_type'] == LinkType.GROUND_GRID else C_VACUUM
        return d['distance']# / c

    def get_udl_path(self, ovl):
        udl_path = []
        for node1, node2 in zip(ovl[:-1], ovl[1:]):
            try:
                sp = nx.shortest_path(self.graph, source=node1, target=node2, weight=self._delay_weight)
            except nx.NetworkXNoPath:
                return None
            udl_path+=sp[:-1]
        udl_path.append(ovl[-1])
        return(udl_path)

    def get_ovls(self, source, target, n_ovls, time):
        ovls = []
        ac_sat_connections = self.graph.neighbors(source)

        for con in ac_sat_connections:
            if self.graph.nodes[con]['node_type'] == 'satellite':
                for gw in self.gw_nodes:
                    path = [source, con, gw[0], target]
                    (dist, _) = self._get_path_distance_delay(path, time)
                    ovls.append({'overlay_path': path, 'id': f'{source}-{con}-{gw[0]}-{target}', 'dist': dist})
            elif self.graph.nodes[con]['node_type'] == 'aviation':
                path = [source, con, target]
                (dist, _) = self._get_path_distance_delay(path, time)
                ovls.append({'overlay_path': path, 'id': f'{source}-{con}-{target}', 'dist': dist})

        best_ovls = sorted(ovls, key=lambda o: o['dist'])[:n_ovls]
        return best_ovls

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
    def slant_range_m(sat_alt_m, elevation_deg, re_m=6_371_000.0):
        el = np.radians(elevation_deg)
        r = re_m + sat_alt_m
        return np.sqrt(r**2 - (re_m * np.cos(el))**2) - re_m * np.sin(el)

    @staticmethod
    def is_in_europe(lat, lon):
        # Rough bounding box for Europe
        return 20 <= lat <= 70 and -60 <= lon <= 40 # Includes North Atlantic

    @staticmethod
    def _dist_m(a_pos, b_pos) -> float:
        return float(np.linalg.norm((a_pos - b_pos).to(unit.m).value))
    
    @staticmethod
    def _node_id(sat) -> str:
        return f'{sat.plane}_{sat.index}'