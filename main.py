import math
import copy
import torch
import os

from algorithm.predictor.encoder_util import snapshot_to_pyg
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
from analytics.overlay_analytics import analyse_overlay_delays
from analytics.queue_delay import test_queue_delay_dynamics

# --- Initialise Ground Network ---
gw_graph      = build_gateway_graph(get_gateways())
aviation_graph = build_aviation_graph(generate_aviation_nodes(30))
upf_graph     = build_upf_graph(get_upfs())
static_ground = build_static_ground_graph(gw_graph, aviation_graph, upf_graph)

# --- Initialise Satellite Constellation ---
walker = WalkerDelta(T, P, F, INC, SATELLITE_ALTITUDE)
sats = walker.generate()



#----------------------------------------
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
dataset_path = os.path.abspath(dataset_path)

# Ensure data dir exists
data_dir = os.path.dirname(dataset_path)
os.makedirs(data_dir, exist_ok=True)

# --- Initialise Aircraft ---
aircraft1 = Aircraft((51,-50), (41,12), 900, 'aircraft1') #Aircraft 900 km/h from North Atlantic Ocean to Rome
aircraft2 = Aircraft((48,-45), (10,12), 900, 'aircraft2')
aircraft3 = Aircraft((48,-20), (30,0), 900, 'aircraft3') 
aircrafts = [aircraft1, aircraft2, aircraft3]

# --- Initialise Dynamic Network ---
current_time = Time("2026-10-08 14:30:00", scale="utc") #Time.now()
network = DynamicNetwork(sats, P, math.floor(T / P), F, current_time, static_ground, aircrafts)

# Simulation parameters
t = 0.0
t_end = SIM_DURATION
dt = TIME_STEP

# Test node-based delay
test_queue_delay_dynamics(network, dt=TIME_STEP)

euro_graph = network.euro_graph
euro_graph_list = []
time_list = [current_time]
overlay_records = []
history = {}

while t < t_end:
    tstep = t//dt
    euro_graph_list.append({'t': t, 'graph': copy.deepcopy(euro_graph)})

    #data = snapshot_to_pyg(euro_graph)
    #history[tstep] = {}
    #history[tstep]['t'] = t
    #history[tstep] = {}
    #history[tstep]['x'] = data.x
    #history[tstep]['edge_index'] = data.edge_index
    #history[tstep]['edge_attr'] = data.edge_attr
    #history[tstep]['node_names'] = data.node_names
    #history[tstep]['name_to_idx'] = data.name_to_idx

    ovls = []
    ovls += network.get_ovls('aircraft1', 'UPF_DE_Frankfurt',3, current_time)
    ovls += network.get_ovls('aircraft2', 'UPF_DE_Frankfurt', 3, current_time) 
    ovls += network.get_ovls('aircraft3', 'UPF_DE_Frankfurt', 3, current_time)  
    print(f'Time is {t}')
    for ovl in ovls:
        udl = network.get_udl_path(ovl['overlay_path'])
        if udl:
            delay_ms = network.get_udl_path_delay(udl, current_time)*1000
            overlay_records.append({'t': t, 'overlay': ovl['overlay_path'],
                                    'udl': udl, 'delay_ms': delay_ms})
            print(f'udl {udl} -- {network.get_udl_path_delay(udl, current_time)*1000} ms delay at time {t}')

    #history[tstep]['overlay_paths'] = overlay_records
    (_, euro_graph) = network.update(dt)
    current_time += TimeDelta(dt, format="sec")
    time_list.append(current_time)
    t += dt

#torch.save({'history_list': history}, dataset_path)
print("Simulation complete. Generating visualization...")
# --- plotting ---
plot_full_graph_timeline([euro_graph_list[i]['graph'] for i in range(len(euro_graph_list))], time_list, title="European Constellation Over Time")
#plot_full_graph_timeline(graph_list, time_list, title="Full Constellation Over Time")

analyse_overlay_delays(overlay_records, dt)