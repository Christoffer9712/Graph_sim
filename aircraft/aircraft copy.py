import numpy as np
import networkx as nx
import ML
from dataclasses import dataclass
from enum import IntEnum

@dataclass(frozen=True)
class TrafficDescription:
    fiveQI:   int
    BW:       float
    UPF:      str

class LinkType(IntEnum):
    SA2A  = 1   # Type 1 — satellite gateway
    DA2G  = 2   # Type 2 — aviation network node

@dataclass(frozen=True)
class TunnelDescription:
    fiveQI:   int
    BW:       float
    linkType: LinkType
    firstHop: str
    GW:       str
    UPF:      str

class Aircraft:
    def __init__(self, startPos, destPot, speed, node_id):
        self.startPos = startPos
        self.destPot = destPot
        self.vel = speed*(destPot - startPos) / np.linalg.norm(destPot - startPos)
        self.position = startPos
        self.node_id = node_id
        self.graph = nx.Graph()
        self.graph.add_node(node_id, node_type='aircraft', position=self.position)

    def propagate(self, dt):
        self.position += self.vel * dt
        self.graph.nodes[self.node_id]['position'] = self.position


    def setTrafficDemand(self, trafficDemand : {TrafficDescription}):
        self.trafficDemand = trafficDemand

    def setUpTunnels(self, dt_tunnel : float, graph : nx.Graph):
        self.tunnels = ML.get_tunnels(self.trafficDemand, dt_tunnel, graph)
    
    def sendData(self, traffic : {TrafficDescription}, graph : nx.Graph):
        PER_list = []
        latency_list = []
        for desc in traffic:
            tunnel = self.mapToTunnel(desc)
            path = self.getPath(tunnel, graph)
            (PER, latency) = self.getPathMetrics(path, graph)
            PER_list.append(PER)
            latency_list.append(latency)
        return (PER_list, latency_list)
    

    def mapToTunnel(self, desc : TrafficDescription):
        for tunnel in self.tunnels:
            if tunnel.fiveQI == desc.fiveQI and tunnel.BW >= desc.BW: #should check available bandwidth!
                return tunnel
        return None

    def getPath(self, tunnel, graph):
        path = nx.shortest_path(graph, self.node_id, tunnel.firstHop)
        if tunnel.linkType == LinkType.DA2G:
            destNode = tunnel.UPF
            path += nx.shortest_path(graph, tunnel.firstHop, tunnel.UPF)
        elif tunnel.linkType == LinkType.SA2A:
            destNode = tunnel.GW
            path += nx.shortest_path(graph, tunnel.firstHop, tunnel.GW)
            path += nx.shortest_path(graph, tunnel.GW, tunnel.UPF)
        return path
    
    def getPathMetrics(self, path, graph):
        PER = 1.0
        latency = 0.0
        for i in range(len(path)-1):
            edge_data = graph.get_edge_data(path[i], path[i+1])
            PER += edge_data['per'] #should be more complex than this!
            latency += edge_data['latency'] #should be more complex than this!
        return (PER, latency)
    
