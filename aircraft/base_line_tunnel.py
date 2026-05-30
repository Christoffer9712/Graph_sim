import networkx as nx

from ground_network.nodes import GroundNodeType
from .types import LinkType, TunnelDescription, TrafficDescription
from utils.utils import haversine_m

def _get_closest_da2g_link(graph: nx.Graph, node_id: str, target_node: str) -> str | None:
    da2g_links = [nbr for nbr in graph.neighbors(node_id) if graph.edges[node_id, nbr]['link_type'] == LinkType.DA2G]
    if not da2g_links:
        return None
    closest_link = min(da2g_links, key=lambda nbr: haversine_m([graph.nodes[node_id]['lat'], graph.nodes[node_id]['lon']], [graph.nodes[nbr]['lat'], graph.nodes[nbr]['lon']]))
       
    return closest_link

def _get_closest_sa2a_link(graph: nx.Graph, node_id: str, target_node: str) -> str | None:
    sa2a_links = [nbr for nbr in graph.neighbors(node_id) if graph.edges[node_id, nbr]['link_type'] == LinkType.SA2A]
    if not sa2a_links:
        return None
   
    return sa2a_links[0]

def _get_closest_sa2a_gw(graph: nx.Graph, sa2a_link: str, target_node: str, gws: list[str]) -> str | None:
    gw = min(gws, key=lambda gw: haversine_m([graph.nodes[target_node]['lat'], graph.nodes[target_node]['lon']], [graph.nodes[gw]['lat'], graph.nodes[gw]['lon']]))
    return gw

def _get_closest_upf(graph: nx.Graph, node_id: str, dn: str) -> str | None:
    upfs = [upf for upf in graph.nodes() if graph.nodes[upf]['node_type'] == GroundNodeType.UPF and dn in graph.nodes[upf]['dn']]
    print(f"Found UPFs for DN {dn}: {upfs}")
    if not upfs:
        return None
    return min(upfs, key=lambda upf: haversine_m([graph.nodes[node_id]['lat'], graph.nodes[node_id]['lon']], [graph.nodes[upf]['lat'], graph.nodes[upf]['lon']]))

def tunnel_setup(aircraft_id: str, dt_tunnel: float, graph: nx.Graph, trafficDemand: dict[int, TrafficDescription]) -> list[TunnelDescription]:
    sorted_traffic_demand = sorted(trafficDemand.values(), key=lambda x: x.fiveQI)
    tunnels = []
    for desc in sorted_traffic_demand:
        upf = _get_closest_upf(graph, aircraft_id, desc.DN)
        if not upf:
            print(f"No UPF found for aircraft {aircraft_id} with DN {desc.DN}. Skipping tunnel setup for this traffic demand.")
            continue
        link = _get_closest_da2g_link(graph, aircraft_id, upf)
        gw = link
        linkType=LinkType.DA2G
        if not link:
            print(f"No DA2G link available for aircraft {aircraft_id} to UPF {upf}, trying SA2A...")
            link = _get_closest_sa2a_link(graph, aircraft_id, upf)
            if not link:
                print(f"No SA2A link available for aircraft {aircraft_id} to UPF {upf}. Skipping tunnel setup for this traffic demand.")
                continue
            gw = _get_closest_sa2a_gw(graph, link, upf, [gw for gw in graph.nodes if graph.nodes[gw]['node_type'] == GroundNodeType.GATEWAY]) # Should not loop over all nodes at each step!
            linkType=LinkType.SA2A
        
        if not gw:
            print(f"No gateway found for SA2A link {link} to UPF {upf}. Skipping tunnel setup for this traffic demand.")
            continue

        tunnels.append(TunnelDescription(
            fiveQI=desc.fiveQI,
            BW=desc.BW,
            linkType=linkType,  # Placeholder; replace with actual link type
            firstHop=link,     # Placeholder; replace with actual first hop
            GW=gw,               # Placeholder; replace with actual GW
            UPF=upf              # Placeholder; replace with actual UPF
        ))
    print(f"Aircraft {aircraft_id} set up tunnels: {tunnels}")
    return tunnels



