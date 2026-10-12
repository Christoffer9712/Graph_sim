import math
import copy

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
#from aircraft.ml import (cost_function, initial_tunnel_cost)
# --- Initialise Ground Network ---

gw_graph      = build_gateway_graph(get_gateways())
aviation_graph = build_aviation_graph(generate_aviation_nodes(30))
upf_graph     = build_upf_graph(get_upfs())
static_ground = build_static_ground_graph(gw_graph, aviation_graph, upf_graph)

# --- Initialise Satellite Constellation ---
walker = WalkerDelta(T, P, F, INC, SATELLITE_ALTITUDE)
sats = walker.generate()

# --- Initialise Aircraft ---
aircraft1 = Aircraft((51,-50), (41,12), 900, 'aircraft1') #Aircraft 900 km/h from North Atlantic Ocean to Rome
aircraft2 = Aircraft((48,-45), (41,12), 900, 'aircraft2') #Aircraft 900 km/h from North Atlantic Ocean to Rome
aircrafts = [aircraft1, aircraft2]

# --- Initialise Dynamic Network ---
current_time = Time("2026-10-08 14:30:00", scale="utc") #Time.now()
network = DynamicNetwork(sats, P, math.floor(T / P), F, current_time, static_ground, aircrafts)
propagator = Propagator(sats, aircrafts)

# Simulation parameters
t = 0.0
t_end = SIM_DURATION
dt = TIME_STEP

# For visualization, we store a copy of the graph at every time step.
#graph = network.graph
#graph_list = []

euro_graph = network.euro_graph
euro_graph_list = []

time_list = [current_time]
full_graph_list = []

tunnel_selection_cost = 0
# --- simulation loop ---
while t < t_end:
    #graph_list.append(copy.deepcopy(graph))
    euro_graph_list.append(copy.deepcopy(euro_graph))
    
    propagator.step(TIME_STEP)
    (_, euro_graph) = network.update(dt)
    ovls = network.get_ovls('aircraft1', 'UPF_DE_Frankfurt', 3, current_time) 
    for ovl in ovls:
        udl = network.get_udl_path(ovl['overlay_path'])
        if udl:
            print(f'udl {udl} -- {network.get_udl_path_delay(udl, current_time)*1000} ms delay at time {t}')

    current_time += TimeDelta(dt, format="sec")
    time_list.append(current_time)
    t += dt

#print(full_graph_list[0])
print("Simulation complete. Generating visualization...")
# --- plotting ---
plot_full_graph_timeline(euro_graph_list, time_list, title="European Constellation Over Time")
#plot_full_graph_timeline(graph_list, time_list, title="Full Constellation Over Time")