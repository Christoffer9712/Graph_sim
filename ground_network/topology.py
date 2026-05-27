import networkx as nx
import numpy as np
from utils.coordinates import eci_to_latlon_batch
from .nodes import GroundNodeType
from .utils import elevation_angle_deg, eci_to_altitude
from config import MAX_SAT_PER_GW, LINK_PER

R_EARTH_M  = 6_371_000.0
C_VACUUM   = 3e8     # m/s  — wireless / space links
C_FIBER    = 2e8     # m/s  — ~0.67c in glass fibre

# Maximum slant range for aircraft-to-ground-node links (200 km).
A2G_RANGE_M = 200_000.0


# ─── geometry helpers ─────────────────────────────────────────────────────────

def _haversine_m(latlon_a, latlon_b) -> float:
    """Great-circle distance in metres between two (lat, lon) pairs (degrees) using haversine formula."""
    lat1, lon1 = np.radians(latlon_a)
    lat2, lon2 = np.radians(latlon_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2.0 * R_EARTH_M * np.arcsin(np.sqrt(a))


def _slant_range_m(sat_alt_m: float, elevation_deg: float) -> float:
    """
    Approximate slant range from ground to satellite given altitude and elevation angle.
    Derived from the law of sines on the Earth-centre / ground-station / satellite triangle.
    The flat-Earth approximation sat_alt / sin(el) is used for el > 10° and is accurate
    to within ~2% at 20° elevation for LEO altitudes.
    """
    el_rad = np.radians(max(elevation_deg, 1.0))   # clamp avoids division by zero
    return sat_alt_m / np.sin(el_rad)


# ─── static ground graph ──────────────────────────────────────────────────────

def build_static_ground_graph(
    gw_graph:       nx.Graph,
    aviation_graph: nx.Graph,
    upf_graph:      nx.Graph,
) -> nx.Graph:
    """
    Merge the three ground-layer graphs and wire the nearest-neighbour ground grid.
    GW↔satellite and aircraft links are dynamic and added in build_full_graph.
    """
    G = nx.Graph()
    G.update(gw_graph)
    G.update(aviation_graph)
    G.update(upf_graph)
    _connect_ground_nodes_as_grid(G, max_neighbors=4)
    return G


def _edge_type_for_ground_pair(data_u: dict, data_v: dict) -> str:
    types = {data_u['node_type'], data_v['node_type']}
    if types == {GroundNodeType.AVIATION}:
        return 'aviation_aviation'
    if types == {GroundNodeType.GATEWAY}:
        return 'gateway_gateway'
    if types == {GroundNodeType.UPF}:
        return 'upf_upf'
    if types == {GroundNodeType.AVIATION, GroundNodeType.GATEWAY}:
        return 'aviation_gateway'
    if types == {GroundNodeType.AVIATION, GroundNodeType.UPF}:
        return 'aviation_upf'
    if types == {GroundNodeType.GATEWAY, GroundNodeType.UPF}:
        return 'upf_gateway'
    return 'NONE'


def _connect_ground_nodes_as_grid(G: nx.Graph, max_neighbors: int = 4) -> None:
    """
    KNN-style grid connectivity: each node gets at most max_neighbors edges,
    always connecting to the geographically closest available partner first.
    Adds latency and per attributes to every edge.
    """
    ground_nodes = [
        (n, d) for n, d in G.nodes(data=True)
        if d['node_type'] in {
            GroundNodeType.GATEWAY,
            GroundNodeType.AVIATION,
            GroundNodeType.UPF,
        }
    ]
    if len(ground_nodes) < 2:
        return

    node_ids = [n for n, _ in ground_nodes]
    coords   = np.array([[d['lat'], d['lon']] for _, d in ground_nodes])

    pairs = sorted(
        ((_haversine_m(coords[i], coords[j]), i, j)
         for i in range(len(coords))
         for j in range(i + 1, len(coords))),
        key=lambda x: x[0],
    )

    degree = [0] * len(node_ids)
    for dist_m, i, j in pairs:
        if degree[i] >= max_neighbors or degree[j] >= max_neighbors:
            continue
        u, v = node_ids[i], node_ids[j]
        if G.has_edge(u, v):
            continue
        if dist_m >= 1000_000: #1000 km max direct connectivity 
            continue
        #ltype = _edge_type_for_ground_pair(G.nodes[u], G.nodes[v])
        ltype='ground_grid'
        G.add_edge(u, v,
                   link_type= ltype,
                   latency=dist_m / C_FIBER,
                   per=LINK_PER[ltype])
        degree[i] += 1
        degree[j] += 1


# ─── dynamic full graph ───────────────────────────────────────────────────────

def build_full_graph(
    sat_graph:               nx.Graph,
    static_ground:           nx.Graph,
    aircraft:                nx.Graph,
    elevation_threshold_deg: float = 20.0,
    current_time=None,
) -> nx.Graph:
    """
    Assemble the complete topology for one simulation timestep.

    Adds three categories of dynamic edges on top of the static ground graph:
      1. GW ↔ satellite     — elevation-angle gated
      2. Aircraft ↔ satellite (A2S) — same elevation check as GW
      3. Aircraft ↔ aviation node (A2G) — slant-range gated at A2G_RANGE_M
    """
    G = nx.Graph()
    G.update(sat_graph)
    G.update(static_ground)
    G.update(aircraft)

    # ── satellite lat/lon and altitude for this timestep ─────────────────────
    sat_ids   = list(sat_graph.nodes)
    positions = np.stack([sat_graph.nodes[n]['position'] for n in sat_ids])
    latitudes, longitudes = eci_to_latlon_batch(positions, current_time)

    sat_lat = {n: float(latitudes[i])  for i, n in enumerate(sat_ids)}
    sat_lon = {n: float(longitudes[i]) for i, n in enumerate(sat_ids)}
    sat_alt = {n: eci_to_altitude(sat_graph.nodes[n]['position']) for n in sat_ids}

    # ── GW ↔ satellite ────────────────────────────────────────────────────────
    gw_nodes = [
        (n, d) for n, d in static_ground.nodes(data=True)
        if d['node_type'] == GroundNodeType.GATEWAY
    ]

    for gw_id, gw_data in gw_nodes:
        reachableSats = []
        for sat_id in sat_ids:
            el = elevation_angle_deg(
                gw_data['lat'], gw_data['lon'],
                sat_lat[sat_id], sat_lon[sat_id], sat_alt[sat_id],
            )
            if el >= elevation_threshold_deg:
                sr = _slant_range_m(sat_alt[sat_id], el)
                reachableSats.append((sat_id, sr))
        
        
        reachableSats.sort(key=lambda x: x[1])  # Sort by slant range (closest first)
        for sat_id, sr in reachableSats[:MAX_SAT_PER_GW]:
            G.add_edge(gw_id, sat_id,
                        link_type='gw_sat',
                        elevation_deg=round(el, 2),
                        latency=sr / C_VACUUM,
                        per=LINK_PER['gw_sat'])

    # ── aircraft links ────────────────────────────────────────────────────────
    av_nodes = [
        (n, d) for n, d in static_ground.nodes(data=True)
        if d['node_type'] == GroundNodeType.AVIATION
    ]

    for ac_id, ac_data in aircraft.nodes(data=True):
        ac_lat = float(ac_data['lat'])
        ac_lon = float(ac_data['lon'])

        # A2S — same elevation check as GW↔sat
        for sat_id in sat_ids:
            el = elevation_angle_deg(
                ac_lat, ac_lon,
                sat_lat[sat_id], sat_lon[sat_id], sat_alt[sat_id],
            )
            if el >= elevation_threshold_deg:
                sr = _slant_range_m(sat_alt[sat_id], el)
                G.add_edge(ac_id, sat_id,
                           link_type='a2s',
                           elevation_deg=round(el, 2),
                           latency=sr / C_VACUUM,
                           per= LINK_PER['a2s'])

        # A2G — within slant-range threshold (horizontal distance approximation)
        for av_id, av_data in av_nodes:
            dist_m = _haversine_m((ac_lat, ac_lon), (av_data['lat'], av_data['lon']))
            if dist_m <= A2G_RANGE_M:
                G.add_edge(ac_id, av_id,
                           link_type='a2g',
                           latency=dist_m / C_VACUUM,
                           per= LINK_PER['a2g'])

    return G