"""
DynamicNetwork: owns the graph and the simulation clock, and calls the
propagator, topology and queue models in order each step.
"""
import networkx as nx
from astropy.time import Time, TimeDelta

from net_simulation.propagator import Propagator
from net_simulation.topology import TopologyModel
from net_simulation.queues import QueueModel, QueueParams
from net_simulation import routing


class DynamicNetwork:
    def __init__(self, satellites, P: int, S: int, F: int, start_time: Time,
                 ground_network: nx.Graph, aircrafts, queue_params: QueueParams | None = None, seed: int = 42):
        self.start_time = start_time
        self.nbrUpdates = 0

        self.graph = ground_network.to_directed()  # Start with the ground network; satellites, aircraft and links are added to this
        self.propagator = Propagator(satellites, aircrafts, start_time)
        self.topology = TopologyModel(self.graph, satellites, aircrafts, ground_network, P, S)
        self.queues = QueueModel(self.graph, queue_params, seed)
        self.queues.init()                         # after the topology has added all nodes

    # ----Public----
    def update(self, dt) -> tuple[nx.DiGraph, nx.DiGraph]:
        """
        Advance the whole network by one simulation step of dt seconds.
        """
        self.nbrUpdates += 1
        t = self.start_time + TimeDelta(self.nbrUpdates * dt, format="sec")
        self.propagator.step(dt, t)   # where things are
        self.topology.update(dt)      # which links exist
        self.queues.step(dt)          # how loaded the nodes are
        return (self.graph, self.euro_graph)

    @property
    def euro_graph(self):
        """Read-only view of the European part of the network (no copy)."""
        return self.graph.subgraph(self.topology.euro_nodes)

    # ---- Routing (forwarded so main.py does not need to change) ----
    def get_ovls(self, source, target, n_ovls, time=None):
        return routing.get_ovls(self.graph, self.topology.gw_nodes, source, target, n_ovls)

    def get_udl_path(self, ovl):
        return routing.get_udl_path(self.graph, ovl)

    def get_udl_path_delay(self, udl, time=None):
        return routing.get_udl_path_delay(self.graph, udl)