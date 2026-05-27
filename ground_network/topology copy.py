import networkx as nx
import numpy as np
from utils.coordinates import eci_to_latlon_batch
from .nodes import GroundNodeType
from .utils import elevation_angle_deg, eci_to_altitude


def build_static_ground_graph(
    gw_graph:       nx.Graph,
    aviation_graph: nx.Graph,
    upf_graph:      nx.Graph,
) -> nx.Graph:
    """
    Merge the three ground-layer graphs and wire all time-invariant edges:
      UPF → every GW         (link_type='upf_gw')
      UPF → every AV node    (link_type='upf_av')

    GW↔satellite links are intentionally excluded — call build_full_graph
    each timestep to add those.
    """
    G = nx.Graph()
    G.update(gw_graph)
    G.update(aviation_graph)
    G.update(upf_graph)

    _connect_ground_nodes_as_grid(G, max_neighbors=4)
    return G


def _haversine_distance(latlon_a, latlon_b):
    lat1, lon1 = np.radians(latlon_a)
    lat2, lon2 = np.radians(latlon_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * np.arcsin(np.sqrt(a))


def _edge_type_for_ground_pair(node_u, node_v, data_u, data_v):
    if data_u['node_type'] == GroundNodeType.AVIATION and data_v['node_type'] == GroundNodeType.AVIATION:
        return 'aviation_aviation'
    if {
        data_u['node_type'], data_v['node_type']
    } == {GroundNodeType.AVIATION, GroundNodeType.GATEWAY}:
        return 'aviation_gateway'
    return 'ground_grid'


def _connect_ground_nodes_as_grid(G: nx.Graph, max_neighbors: int = 4) -> None:
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
    coords = np.array([[d['lat'], d['lon']] for _, d in ground_nodes])
    pairs = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = _haversine_distance(coords[i], coords[j])
            pairs.append((dist, i, j))
    pairs.sort(key=lambda item: item[0])

    degree = [0] * len(node_ids)
    for _, i, j in pairs:
        if degree[i] >= max_neighbors or degree[j] >= max_neighbors:
            continue
        u, v = node_ids[i], node_ids[j]
        if G.has_edge(u, v):
            continue
        edge_type = _edge_type_for_ground_pair(u, v, G.nodes[u], G.nodes[v])
        G.add_edge(u, v, link_type=edge_type)
        degree[i] += 1
        degree[j] += 1


def build_full_graph(
    sat_graph:              nx.Graph,
    static_ground:          nx.Graph,
    aircraft:               nx.Graph,
    elevation_threshold_deg: float = 20.0,
    current_time = None,
) -> nx.Graph:
    """
    Assemble the complete topology for one simulation timestep.

    Parameters
    ----------
    sat_graph : nx.Graph
        Satellite-only graph. Each node must have a 'position' attribute
        holding its ECI vector — used to derive altitude via eci_to_altitude().
    static_ground : nx.Graph
        Output of build_static_ground_graph (call once, reuse every timestep).
    aircraft:               nx.Graph
        Aircraft graph.
    elevation_threshold_deg : float
        Minimum elevation angle for a GW↔satellite link to be active.

    Returns
    -------
    nx.Graph
        Full graph: satellite nodes + ISLs, ground nodes + static edges,
        and dynamic GW↔satellite edges for this timestep.
    """
    G = nx.Graph()
    G.update(sat_graph)
    G.update(static_ground)
    G.update(aircraft)
    gw_nodes = [
        (n, d) for n, d in static_ground.nodes(data=True)
        if d['node_type'] == GroundNodeType.GATEWAY
    ]


    sat_coords = {}

    positions = np.stack([sat_graph.nodes[node]['position'] for node in sat_graph.nodes])
    latitudes, longitudes = eci_to_latlon_batch(positions, current_time)
    for idx, node in enumerate(sat_graph.nodes):
        sat_coords[node] = (latitudes[idx], longitudes[idx])
       

    for gw_id, gw_data in gw_nodes:
        for sat_id, (sat_lat, sat_lon) in sat_coords.items():
            sat_alt = eci_to_altitude(sat_graph.nodes[sat_id]['position'])
            el = elevation_angle_deg(
                gw_data['lat'], gw_data['lon'],
                sat_lat, sat_lon, sat_alt,
            )
            if sat_id == '2-19':
                print(f"Checking link {gw_id} ↔ {sat_id} at time {current_time}:")
                print(f"  Satellite at lat={sat_lat:.2f}, lon={sat_lon:.2f}, alt={sat_alt:.2f} m, elevation angle={el:.2f}°")
                print(f"  Satellite position: {sat_graph.nodes[sat_id]['position']}")
            if el >= elevation_threshold_deg:
                G.add_edge(gw_id, sat_id, link_type='gw_sat', elevation_deg=round(el, 2))

    return G