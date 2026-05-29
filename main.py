import math

from astropy.time import Time, TimeDelta
from config import *
from orbit.constellation import WalkerDelta
from net_simulation.propagator import Propagator
from net_simulation.network import DynamicNetwork
from utils.visualization import *
from ground_network import (
    get_gateways, generate_aviation_nodes, get_upfs,
    build_gateway_graph, build_aviation_graph, build_upf_graph,
    build_static_ground_graph
)
from aircraft.aircraft import *
from aircraft.types import TrafficDescription
import astropy.units as u
import copy

# --- Initialise Ground Network ---
gw_graph      = build_gateway_graph(get_gateways())
aviation_graph = build_aviation_graph(generate_aviation_nodes(30))
upf_graph     = build_upf_graph(get_upfs())
static_ground = build_static_ground_graph(gw_graph, aviation_graph, upf_graph)

# --- Initialise Satellite Constellation ---
walker = WalkerDelta(T, P, F, INC, SATELLITE_ALTITUDE)
sats = walker.generate()

# --- Initialise Aircraft ---
aircraft = Aircraft((51,0), (41,12), 900, 'aircraft') #Aircraft 900 km/h from London to Rome
demands = [
    TrafficDescription(fiveQI=1, BW=5.0,  UPF='UPF_DE_Frankfurt'),  # URLLC control (later, don't hardcode UPF but use DN)
    TrafficDescription(fiveQI=5, BW=50.0, UPF='UPF_NL_Amsterdam'),  # eMBB video
]
aircraft.setTrafficDemand(demands) # Constant for now

# --- Initialise Dynamic Network ---
current_time = Time.now()
only_europe = True
network = DynamicNetwork(sats, P, math.floor(T / P), F, current_time, static_ground, [aircraft], only_europe)
propagator = Propagator(sats, [aircraft])

# Simulation parameters
t = 0.0
t_end = SIM_DURATION
dt = float(TIME_STEP.to_value(u.s))
tunnel_update_interval = TUNNELING_DECISION_INTERVAL
update_step = tunnel_update_interval // dt

# For visualization, we store a copy of the graph at every time step.
graph = network.graph
graph_list = []

euro_graph = network.euro_graph
euro_graph_list = []

time_list = [current_time]
full_graph_list = []

# --- simulation loop ---
while t < t_end:
    graph_list.append(copy.deepcopy(graph))
    euro_graph_list.append(copy.deepcopy(euro_graph))
    
   # if t % update_interval < dt or t < dt:
   #      print(f"Time {t:.1f}s: Updating tunnels for aircraft {aircraft.node_id}...")
   #      aircraft.setUpTunnels(update_interval, euro_graph)

    #(PER, latency) = aircraft.sendData(demands, euro_graph)
    #print(f"Time {t:.1f}s: Aircraft at {aircraft.position}, PER={PER}, latency={latency}s")
    
    #if t % update_interval < dt or t < dt:
    #    agent.observe(full_graph, aircraft)
    #    agent.act(aircraft)

    propagator.step(TIME_STEP)
    (graph, euro_graph) = network.update(dt)
    current_time += TimeDelta(dt, format="sec")
    time_list.append(current_time)
    t += dt

#print(full_graph_list[0])
print("Simulation complete. Generating visualization...")
# --- plotting ---
plot_full_graph_timeline(euro_graph_list, time_list, title="European Constellation Over Time")
plot_full_graph_timeline(graph_list, time_list, title="Full Constellation Over Time")