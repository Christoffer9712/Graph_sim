import networkx as nx
import numpy as np
from .nodes import GroundNodeType
from config import LINK_PER, C_FIBER, LinkType
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
    _connect_ground_nodes_as_grid(G, max_neighbors=4)
    return G


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
        if dist_m >= 1000_000: #1000 km max direct connectivity 
            continue
        #ltype = _edge_type_for_ground_pair(G.nodes[u], G.nodes[v])
        ltype=LinkType.GROUND_GRID
        G.add_edge(u, v,
                   link_type= ltype,
                   distance=dist_m,
        )
        degree[i] += 1
        degree[j] += 1