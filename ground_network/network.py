import networkx as nx
from .nodes import GroundNode


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