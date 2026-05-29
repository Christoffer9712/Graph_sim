import networkx as nx
from .types import LinkType, TunnelDescription, TrafficDescription
from utils.utils import haversine_m

def _get_closest_da2g_link(graph: nx.Graph, node_id: str, target_node: str) -> str:
    da2g_links = [nbr for nbr in graph.neighbors(node_id) if graph.edges[node_id, nbr]['link_type'] == LinkType.DA2G]
    if not da2g_links:
        raise ValueError(f"No DA2G links found for node {node_id}")
    closest_link = min(da2g_links, key=lambda nbr: haversine_m([graph.nodes[node_id]['lat'], graph.nodes[node_id]['lon']], [graph.nodes[nbr]['lat'], graph.nodes[nbr]['lon']]))
        
        
    return closest_link


def tunnel_setup(aircraft_id: str, dt_tunnel: float, graph: nx.Graph, trafficDemand: list[TrafficDescription]) -> list[TunnelDescription]:
    sorted_traffic_demand = sorted(trafficDemand, key=lambda x: x.fiveQI) 
    tunnels = []
    for desc in sorted_traffic_demand:
        closest_link = _get_closest_da2g_link(graph, aircraft_id, desc.UPF)
        tunnels.append(TunnelDescription(
            fiveQI=desc.fiveQI,
            BW=desc.BW,
            linkType=LinkType.DA2G,  # Placeholder; replace with actual link type
            firstHop=closest_link,     # Placeholder; replace with actual first hop
            GW=closest_link,               # Placeholder; replace with actual GW
            UPF=desc.UPF              # Placeholder; replace with actual UPF
        ))

    return tunnels



