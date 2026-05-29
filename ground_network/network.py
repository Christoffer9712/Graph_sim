import networkx as nx
from .nodes import GroundNode
import numpy as np
from .nodes import GroundNodeType
from config import MAX_GROUND_LINK_DISTANCE, MAX_GROUND_NEIGHBORS, LinkType
from utils.utils import haversine_m

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
    _connect_ground_nodes_as_grid(G, max_neighbors=MAX_GROUND_NEIGHBORS)
    return G


def _connect_ground_nodes_as_grid(G: nx.Graph, max_neighbors: int) -> None:
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
        ((haversine_m(coords[i], coords[j]), i, j)
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
        if dist_m >= MAX_GROUND_LINK_DISTANCE: 
            continue
        ltype=LinkType.GROUND_GRID
        G.add_edge(u, v,
                   link_type = ltype,
                   distance = dist_m,
        )
        degree[i] += 1
        degree[j] += 1

def _add_node(G: nx.Graph, node: GroundNode) -> None:
    G.add_node(
        node.node_id,
        node_type=node.node_type,
        lat=node.lat,
        lon=node.lon,
    )


def build_gateway_graph(gateways: list[GroundNode]) -> nx.Graph:
    """
    One node per GW, no edges.
    GW↔satellite edges are added dynamically in topology.py because
    they depend on satellite positions at each timestep.
    """
    G = nx.Graph()
    for gw in gateways:
        _add_node(G, gw)
    return G


def build_aviation_graph(aviation_nodes: list[GroundNode]) -> nx.Graph:
    """
    One node per aviation node, no edges.
    Aviation nodes only connect upward (to aircraft) and downward (to UPFs).
    """
    G = nx.Graph()
    for node in aviation_nodes:
        _add_node(G, node)
    return G


def build_upf_graph(upfs: list[GroundNode]) -> nx.Graph:
    """One node per UPF, no edges."""
    G = nx.Graph()
    for upf in upfs:
        _add_node(G, upf)
    return G