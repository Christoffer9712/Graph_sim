import networkx as nx
import numpy as np
import astropy.units as unit
from astropy.time import TimeDelta
from astropy.coordinates import EarthLocation

from aircraft import aircraft
from ground_network.nodes import GroundNodeType
from utils.coordinates import eci_to_latlon_batch
from utils.network_link_characteristics import sample_per, markov_step
from utils.utils import haversine_m
from config import (
    MAX_ISL_LINK_DISTANCE, SATELLITE_ALTITUDE, LINK_PARAMS, MAX_SAT_PER_GW, MAX_SAT_PER_AC,
    MAX_A2G_LINK_DISTANCE, LinkType, ALWAYS_UP, C_VACUUM, AIRCRRAFT_ALTITUDE
)


class DynamicNetwork:
    """
    Maintains a NetworkX graph of the Walker-Delta constellation.

    Positions are stored internally as plain numpy arrays in meters so that
    no Astropy Quantity ever enters the graph or the link models.

    Edge attributes
    ───────────────
    distance  (float m)
    link_type (LinkType)
    state     ('UP' | 'DOWN')
    per       (float)
    """
    def __init__(self, satellites, P: int, S: int, F: int, start_time, ground_network: nx.Graph, aircrafts: list, only_europe=False):
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
        feeder_link_elevation_threshold_deg = 20.0
        self.max_feeder_link_dist = self.slant_range_m(float(SATELLITE_ALTITUDE.to_value(unit.m)), feeder_link_elevation_threshold_deg)
        self.max_sa2a_link_dist = self.max_feeder_link_dist  # Assume same for now
        self.max_da2g_link_dist = MAX_A2G_LINK_DISTANCE

        self.graph = ground_network.copy()  # Start with the ground network; we'll add satellites and ISLs to this
        self.euro_graph = nx.Graph()  # Subgraph view containing only European satellites and all ground nodes
        self._euro_nodes: set[str] = set()
        self.nbrUpdates = 0
        self._init_nodes(self.start_time)
        self._init_edges(self.start_time)
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
        t = self.start_time + TimeDelta(self.nbrUpdates * dt, format="sec")
        self._refresh_positions(t)
        self._update_existing_edges(dt,t)
        self._add_candidate_edges(t)

        return (self.graph, self.euro_graph)

    # ──────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────

    @staticmethod
    def _node_id(sat) -> str:
        return f'{sat.plane}-{sat.index}'

    def _init_nodes(self, time):
        for sat in self.satellites:
            nid = self._node_id(sat)
            self._id_to_sat[nid] = sat
            self.graph.add_node(nid, node_type='satellite', position=sat.position)

        for aircraft in self.aircrafts:
            lat, lon = aircraft.position
            eci_pos = self.gs_to_eci(lat, lon, AIRCRRAFT_ALTITUDE, time) # Assume aircraft is at 10 km altitude for ECI conversion
            self.graph.add_node(aircraft.node_id,
                                node_type='aircraft',
                                position=eci_pos,
                                lat=aircraft.position[0],
                                lon=aircraft.position[1])

        positions = np.stack([sat.position.to(unit.m).value for sat in self.satellites]) * unit.m
        lat, lon = eci_to_latlon_batch(positions, time)
        self._euro_nodes = {
            self._node_id(sat)
            for sat, lat_i, lon_i in zip(self.satellites, lat, lon)
            if self.is_in_europe(lat_i, lon_i)
        }.union({n for n in self.ground_network.nodes()}).union(self.aircraft_node_ids)
        self.euro_graph = self.graph.subgraph(self._euro_nodes).copy()

    def _init_edges(self, time):
        sat_euro_ids = []
        for sat in self.satellites:
            u = self._node_id(sat)
            if u in self._euro_nodes:
                sat_euro_ids.append(u)
            for neighbor, link_type in self._topology_neighbors(sat):
                v = self._node_id(neighbor)
                if self.only_europe and (u not in self._euro_nodes or v not in self._euro_nodes):
                    continue
                if self.graph.has_edge(u, v):
                    continue
                dist = self._dist_m(sat.position, neighbor.position)
                if dist <= MAX_ISL_LINK_DISTANCE:
                    self._add_edge(u, v, dist, link_type, state='UP')

        # ── GW ↔ satellite ────────────────────────────────────────────────────────
        self.gw_nodes = [
            (n, d) for n, d in self.ground_network.nodes(data=True)
            if d['node_type'] == GroundNodeType.GATEWAY
        ]
        
        for gw_id, gw_data in self.gw_nodes:
            reachableSats = []
            for sat_id in sat_euro_ids: # only European satellites since GW are in Europe
                gw_pos = self.gs_to_eci(gw_data['lat'], gw_data['lon'], 0, time)

                distance_m = self._dist_m(self.graph.nodes[sat_id]["position"], gw_pos)
                if distance_m <= self.max_feeder_link_dist:
                    reachableSats.append((sat_id, distance_m))

            reachableSats.sort(key=lambda x: x[1])  # Sort by distance (closest first)
            for sat_id, dist in reachableSats[:MAX_SAT_PER_GW]:
                self._add_edge(gw_id, sat_id, dist, LinkType.FEEDER_LINK, state='UP')

        # ──────────────────────────────────────────
        # Aircraft - satellite and aircraft -ground
        self.av_nodes = [
            (n, d) for n, d in self.ground_network.nodes(data=True)
            if d['node_type'] == GroundNodeType.AVIATION
        ]
        
        for aircraft in self.aircrafts:
            reachableSats = []
            ac_id = aircraft.node_id

            # A2S — same elevation check as GW↔sat
            for sat_id in sat_euro_ids:
                distance_m = self._dist_m(self.graph.nodes[sat_id]["position"], self.graph.nodes[aircraft.node_id]["position"])
                if distance_m <= self.max_sa2a_link_dist:
                    reachableSats.append((sat_id, distance_m))

            reachableSats.sort(key=lambda x: x[1])  # Sort by distance (closest first)
            for sat_id, dist in reachableSats[:MAX_SAT_PER_AC]:
                self._add_edge(aircraft.node_id, sat_id, dist, LinkType.SA2A, state='UP')

            # A2G — within slant-range threshold (horizontal distance approximation)
            for av_id, av_data in self.av_nodes:
                dist_m = haversine_m((aircraft.position[0], aircraft.position[1]), (av_data['lat'], av_data['lon']))
                if dist_m <= self.max_da2g_link_dist:
                    self._add_edge(aircraft.node_id, av_id, dist_m, LinkType.DA2G, state='UP')


    # ──────────────────────────────────────────
    # Update helpers
    # ──────────────────────────────────────────

    def _refresh_positions(self, time):
        for sat in self.satellites:
            # Strip units on every refresh
            self.graph.nodes[self._node_id(sat)]['position'] = sat.position

        for aircraft in self.aircrafts:
            lat, lon = aircraft.position
            eci_pos = self.gs_to_eci(lat, lon, AIRCRRAFT_ALTITUDE, time) # Assume aircraft is at 10 km altitude for ECI conversion
            self.graph.nodes[aircraft.node_id]['position'] = eci_pos
            self.graph.nodes[aircraft.node_id]['lat'] = lat
            self.graph.nodes[aircraft.node_id]['lon'] = lon

        if self.only_europe:
            positions = np.stack([sat.position.to(unit.m).value for sat in self.satellites]) * unit.m
            lat, lon = eci_to_latlon_batch(positions, time)
            self._euro_nodes = {
                self._node_id(sat)
                for sat, lat_i, lon_i in zip(self.satellites, lat, lon)
                if self.is_in_europe(lat_i, lon_i)
            }.union({n for n in self.ground_network.nodes()}).union(self.aircraft_node_ids)
            self.euro_graph = self.graph.subgraph(self._euro_nodes).copy()


    def _update_existing_edges(self, dt: float, time):
        to_remove = []

        for u, v, data in self.graph.edges(data=True):
            if self.only_europe and (u not in self._euro_nodes or v not in self._euro_nodes):
                to_remove.append((u, v))
                continue

            if data['link_type'] == LinkType.INTER_PLANE_ISL:
                dist_m = self._dist_m(self.graph.nodes[u]["position"], self.graph.nodes[v]["position"])
                if dist_m > MAX_ISL_LINK_DISTANCE:
                    to_remove.append((u, v))
                    continue

            elif data['link_type'] == LinkType.INTRA_PLANE_ISL:
                dist_m = data['distance']
                if dist_m > MAX_ISL_LINK_DISTANCE:
                    to_remove.append((u, v))
                    continue

            elif data['link_type'] == LinkType.FEEDER_LINK:
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
            
            elif data['link_type'] == LinkType.GROUND_GRID:
                dist_m = data['distance'] # Fixed distance

            elif data['link_type'] == LinkType.SA2A:
                dist_m = self._dist_m(self.graph.nodes[u]["position"], self.graph.nodes[v]["position"])
                if dist_m > self.max_sa2a_link_dist:
                    to_remove.append((u, v))
                    continue
            
            elif data['link_type'] == LinkType.DA2G:
                dist_m = haversine_m((self.graph.nodes[u]['lat'], self.graph.nodes[u]['lon']), (self.graph.nodes[v]['lat'], self.graph.nodes[v]['lon']))
                if dist_m > self.max_da2g_link_dist:
                    to_remove.append((u, v))
                    continue

            else:
                print(f"Unknown link type {data['link_type']} for edge {u} ↔ {v}")
                

            data['distance'] = dist_m
            params = LINK_PARAMS[data['link_type']]
            if ALWAYS_UP:
                data['state'] = 'UP'
            else:
                data['state'] = markov_step(data['state'], params, dist_m, dt)   # 'UP' or 'DOWN'

            
        self.graph.remove_edges_from(to_remove)

    def _add_candidate_edges(self, time):
        sat_euro_ids = set()
        for sat in self.satellites:
            u = self._node_id(sat)
            for neighbor, link_type in self._topology_neighbors(sat):
                v = self._node_id(neighbor)
                if u in self._euro_nodes: # Only consider new edges between European satellites
                    sat_euro_ids.add(u)
                else:
                    continue
                if v in self._euro_nodes:
                    sat_euro_ids.add(v)
                else:                    
                    continue
                if self.graph.has_edge(u, v):
                    continue
                dist = self._dist_m(sat.position, neighbor.position)
                if dist <= MAX_ISL_LINK_DISTANCE: # This assumes new links will always be up (irrespective of MTTF). Mostly problematic for European subgraph where links can come in and out of range frequently
                    self._add_edge(u, v, dist, link_type, state='UP')

        # ── GW ↔ satellite ────────────────────────────────────────────────────────
        for gw_id, gw_data in self.gw_nodes:
            nbr_of_links = sum(1 for _, _, d in self.graph.edges(gw_id, data=True) if d['link_type'] == LinkType.FEEDER_LINK)
            reachableSats = []
            for sat_id in sat_euro_ids: # only European satellites since GW are in Europe
                gw_pos = self.gs_to_eci(gw_data['lat'], gw_data['lon'], 0, time)

                distance_m = self._dist_m(self.graph.nodes[sat_id]["position"], gw_pos)
                if distance_m <= self.max_feeder_link_dist:
                    reachableSats.append((sat_id, distance_m))

            reachableSats.sort(key=lambda x: x[1])  # Sort by distance (closest first)
            for sat_id, dist in reachableSats[:MAX_SAT_PER_GW-nbr_of_links]:
                self._add_edge(gw_id, sat_id, dist, LinkType.FEEDER_LINK, state='UP')

        # ──────────────────────────────────────────
        for aircraft in self.aircrafts:
            # ──────────────────────────────────────────
            # Aircraft - satellite and aircraft -ground
   
            reachableSats = []
            for aircraft in self.aircrafts:
                ac_id = aircraft.node_id

                # A2S — same elevation check as GW↔sat
                for sat_id in sat_euro_ids:
                    distance_m = self._dist_m(self.graph.nodes[sat_id]["position"], self.graph.nodes[aircraft.node_id]["position"])
                    if distance_m <= self.max_sa2a_link_dist:
                        reachableSats.append((sat_id, distance_m))

                    reachableSats.sort(key=lambda x: x[1])  # Sort by distance (closest first)
                    to_remove = [(aircraft.node_id, nbr) for nbr in self.graph.neighbors(aircraft.node_id) if self.graph.edges[aircraft.node_id, nbr]['link_type'] == LinkType.SA2A and nbr not in {sat_id for sat_id, _ in reachableSats[:MAX_SAT_PER_AC]}]
                    to_add = [sat_id for sat_id, _ in reachableSats[:MAX_SAT_PER_AC] if not self.graph.has_edge(aircraft.node_id, sat_id)]
                    self.graph.remove_edges_from(to_remove)
                    for sat_id, dist in (reachableSats[:MAX_SAT_PER_AC]):
                        if sat_id in to_add:
                            self._add_edge(aircraft.node_id, sat_id, dist, LinkType.SA2A, state='UP')


                # A2G — within slant-range threshold (horizontal distance approximation)
                for av_id, av_data in self.av_nodes:
                    dist_m = haversine_m((aircraft.position[0], aircraft.position[1]), (av_data['lat'], av_data['lon']))
                    if dist_m <= self.max_da2g_link_dist:
                        self._add_edge(aircraft.node_id, av_id, dist_m, LinkType.DA2G, state='UP')
                        
        self.euro_graph = self.graph.subgraph(self._euro_nodes).copy()
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
            key=lambda neighbor: self._dist_m(sat.position, neighbor.position)
        )

        return [
            (intra_neighbor, LinkType.INTRA_PLANE_ISL),
            (inter_neighbor, LinkType.INTER_PLANE_ISL),
        ]

    def _add_edge(self, u: str, v: str, distance_m,
                  link_type: LinkType, state: str = 'UP'):
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

    # ─── geometry helpers ─────────────────────────────────────────────────────────
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