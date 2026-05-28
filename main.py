import math

from astropy.time import Time, TimeDelta
from config import *
from orbit.constellation import WalkerDelta
from net_simulation.propagator import Propagator
from net_simulation.network import DynamicNetwork
from utils.coordinates import eci_to_latlon
from utils.visualization import *
from ground_network import (
    get_gateways, generate_aviation_nodes, get_upfs,
    build_gateway_graph, build_aviation_graph, build_upf_graph,
    build_static_ground_graph
)
from e2e_network.topology import build_full_graph
from aircraft.aircraft import *
from aircraft.types import TrafficDescription
#from aircraft.ml import get_agent
import astropy.units as u
import copy

# --- setup ---
gw_graph      = build_gateway_graph(get_gateways())
aviation_graph = build_aviation_graph(generate_aviation_nodes(30))
upf_graph     = build_upf_graph(get_upfs())
static_ground = build_static_ground_graph(gw_graph, aviation_graph, upf_graph)

walker = WalkerDelta(T, P, F, INC, ALTITUDE)
sats = walker.generate()

current_time = Time.now()

only_europe = True

network = DynamicNetwork(sats, MAX_LINK_DISTANCE, P, math.floor(T / P), F, current_time, static_ground, only_europe)

t = 0.0
t_end = SIM_DURATION.to_value(u.s)
dt = float(TIME_STEP.to_value(u.s))

graph = network.graph
graph_list = []
euro_graph = network.euro_graph
euro_graph_list = []
time_list = [current_time]
full_graph_list = []

aircraft = Aircraft((51,0), (41,12), 900, 'aircraft') #Aircraft 900 km/h from London to Rome
demands = [
    TrafficDescription(fiveQI=1, BW=5.0,  UPF='UPF_DE_Frankfurt'),  # URLLC control (later, don't hardcode UPF but use DN)
    TrafficDescription(fiveQI=5, BW=50.0, UPF='UPF_NL_Amsterdam'),  # eMBB video
]
aircraft.setTrafficDemand(demands) # Constant for now

propagator = Propagator(sats, [aircraft])

#agent = get_agent()

update_interval = 100  # how many seconds between each tunneling decision
update_step = update_interval // dt

first = True
# --- simulation loop ---
while t < t_end:
    graph_list.append(copy.deepcopy(graph))
    euro_graph_list.append(copy.deepcopy(euro_graph))

    #full_graph = build_full_graph(
    #    sat_graph     = euro_graph if only_europe else graph,
    #    static_ground = static_ground,
    #    aircraft      = aircraft.graph,
    #    elevation_threshold_deg = 25.0,
    #    current_time = current_time,
    #)
    #full_graph_list.append(full_graph)
    
    #if t % update_interval < dt or t < dt:
    #     print(f"Time {t:.1f}s: Updating tunnels for aircraft {aircraft.node_id}...")
    #     aircraft.setUpTunnels(update_interval, full_graph)

    #(PER, latency) = aircraft.sendData(
    #    demands,
    #    full_graph,
    #)
    #print(f"Time {t:.1f}s: Aircraft at {aircraft.position}, PER={PER}, latency={latency}s")
    
    #if t % update_interval < dt or t < dt:
    #    agent.observe(full_graph, aircraft)
    #    agent.act(aircraft)

    propagator.step(TIME_STEP)
    (graph, euro_graph) = network.update(dt)
    current_time += TimeDelta(dt, format="sec")
    time_list.append(current_time)
    t += dt
    if first:
        for edge in euro_graph.edges(data=True):
            print(f"Edge in Euro graph: {edge}")
        first = False

#print(full_graph_list[0])
print("Simulation complete. Generating visualization...")
# --- plotting ---
plot_full_graph_timeline(euro_graph_list, time_list, title="Full Constellation Over Time")
plot_full_graph_timeline(graph_list, time_list, title="Full Constellation Over Time")
#if only_europe:
#    plot_constellation_timeline(euro_graph_list, time_list, title="European Subgraph Over Time")
#    plot_full_graph_timeline(full_graph_list, time_list, title="Full Graph Timeline")