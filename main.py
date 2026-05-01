import math

from astropy.time import Time, TimeDelta
from config import *
from orbit.constellation import WalkerDelta
from simulation.propagator import Propagator
from simulation.network import SatelliteNetwork
from utils.coordinates import eci_to_latlon
from utils.visualization import *
import astropy.units as u
import copy

# --- setup ---
walker = WalkerDelta(T, P, F, INC, ALTITUDE)
sats = walker.generate()

propagator = Propagator(sats)

current_time = Time.now()
network = SatelliteNetwork(sats, MAX_LINK_DISTANCE, P, math.floor(T / P), F, current_time, only_europe=True)

t = 0.0
t_end = SIM_DURATION.to_value(u.s)
dt = TIME_STEP.to_value(u.s)

graph_list = [network.graph.copy()]
euro_graph_list = [network.euro_graph.copy()]
time_list = [current_time]

# --- simulation loop ---
while t < t_end:

    propagator.step(TIME_STEP)

    (graph, euro_graph) = network.update(dt)
    graph_list.append(copy.deepcopy(graph))
    euro_graph_list.append(copy.deepcopy(euro_graph))
    current_time += TimeDelta(dt, format="sec")
    time_list.append(current_time)
    t += dt

print(len(graph_list), len(time_list))
print(graph_list[0])
print("Simulation complete. Generating visualization...")
# --- plotting ---
plot_constellation_timeline(graph_list, time_list, title="Full Constellation Over Time")
plot_constellation_timeline(euro_graph_list, time_list, title="European Subgraph Over Time")