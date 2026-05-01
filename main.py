import math

from astropy.time import Time, TimeDelta
from config import *
from orbit.constellation import WalkerDelta
from sat_simulation.propagator import Propagator
from sat_simulation.network import SatelliteNetwork
from utils.coordinates import eci_to_latlon
from utils.visualization import *
from ground_network import (
    get_gateways, generate_aviation_nodes, get_upfs,
    build_gateway_graph, build_aviation_graph, build_upf_graph,
    build_static_ground_graph, build_full_graph,
)
import astropy.units as u
import copy

# --- setup ---
# --- one-time setup ---
gw_graph      = build_gateway_graph(get_gateways())
aviation_graph = build_aviation_graph(generate_aviation_nodes(30))
upf_graph     = build_upf_graph(get_upfs())
static_ground = build_static_ground_graph(gw_graph, aviation_graph, upf_graph)

walker = WalkerDelta(T, P, F, INC, ALTITUDE)
sats = walker.generate()

propagator = Propagator(sats)

current_time = Time.now()

only_europe = True

network = SatelliteNetwork(sats, MAX_LINK_DISTANCE, P, math.floor(T / P), F, current_time, only_europe)

t = 0.0
t_end = SIM_DURATION.to_value(u.s)
dt = TIME_STEP.to_value(u.s)

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

    full_graph = build_full_graph(
        sat_graph     = euro_graph if only_europe else graph,
        static_ground = static_ground,
        elevation_threshold_deg = 20.0,
        current_time = current_time,
    )
    full_graph_list.append(full_graph)
    
    propagator.step(TIME_STEP)
    (graph, euro_graph) = network.update(dt)
    current_time += TimeDelta(dt, format="sec")
    time_list.append(current_time)
    t += dt

print(len(full_graph_list), len(time_list))
print(full_graph_list[0])
print("Simulation complete. Generating visualization...")
# --- plotting ---
plot_constellation_timeline(graph_list, time_list, title="Full Constellation Over Time")
if only_europe:
    plot_constellation_timeline(euro_graph_list, time_list, title="European Subgraph Over Time")
    plot_full_graph_timeline(full_graph_list, time_list, title="Full Graph Timeline")