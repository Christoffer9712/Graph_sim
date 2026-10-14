"""
Overlay selection, underlay routing and path delay. Only reads the graph.
"""
import networkx as nx

from config import LinkType, C_VACUUM, C_FIBER
from net_simulation.geometry import dist_m


def get_path_distance_delay(graph, path, calc_delay=False):
    tot_dist = 0
    prop_delay = 0
    for node1, node2 in zip(path[:-1], path[1:]):
        p1 = graph.nodes[node1]['position']
        p2 = graph.nodes[node2]['position']

        dist = dist_m(p1, p2)
        tot_dist += dist
        if calc_delay:
            if graph.get_edge_data(node1, node2)['link_type'] == LinkType.GROUND_GRID:
                prop_delay += dist/C_FIBER
            else:
                prop_delay += dist/C_VACUUM

    return (tot_dist, prop_delay)


def get_udl_path_delay(graph, udl):
    queue_delay = 0
    (_, prop_delay) = get_path_distance_delay(graph, udl, calc_delay=True)
    for node in udl:
        queue_delay += graph.nodes[node]['queue_delay']

    return queue_delay + prop_delay


def delay_weight(u, v, d):
    if d.get('state') != 'UP':
        return None
    #c = C_FIBER if d['link_type'] == LinkType.GROUND_GRID else C_VACUUM
    return d['distance']# / c


def get_udl_path(graph, ovl):
    udl_path = []
    for node1, node2 in zip(ovl[:-1], ovl[1:]):
        n1_type = graph.nodes[node1]['node_type']
        if n1_type == 'aircraft':
            subgraph = graph.subgraph(
                [n for n, d in graph.nodes(data=True) if d['node_type'] in ['satellite', 'aviation', 'aircraft']] + [node2])
        elif n1_type == 'satellite':
            subgraph = graph.subgraph(
                [n for n, d in graph.nodes(data=True) if d['node_type'] == 'satellite'] + [node2])
        else:
            subgraph = graph.subgraph(
                [n for n, d in graph.nodes(data=True) if d['node_type'] not in ['satellite', 'aircraft']])

        try:
            sp = nx.shortest_path(subgraph, source=node1, target=node2, weight=delay_weight)
        except nx.NetworkXNoPath:
            return None
        udl_path += sp[:-1]
    udl_path.append(ovl[-1])
    return udl_path


def get_ovls(graph, gw_nodes, source, target, n_ovls):
    ovls = []
    for con in graph.neighbors(source):
        if graph.nodes[con]['node_type'] == 'satellite':
            for gw in gw_nodes:
                path = [source, con, gw[0], target]
                (dist, _) = get_path_distance_delay(graph, path)
                ovls.append({'ovl_path': path, 'ovl_id': f'{source}-{con}-{gw[0]}-{target}', 'dist': dist})
        elif graph.nodes[con]['node_type'] == 'aviation':
            path = [source, con, target]
            (dist, _) = get_path_distance_delay(graph, path)
            ovls.append({'ovl_path': path, 'ovl_id': f'{source}-{con}-{target}', 'dist': dist})

    return sorted(ovls, key=lambda o: o['dist'])[:n_ovls]