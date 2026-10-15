import math
import torch
import numpy as np

from algorithm.predictor.encoder_util import snapshot_to_pyg
from astropy.time import Time, TimeDelta
from config import *
from orbit.constellation import WalkerDelta
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
from net_simulation.routing import get_udl_prior_delay

def generate_data(dataset_path, start_times=[Time("2026-10-08 14:30:00", scale="utc")], seed=[42], analyse=False, plot=False):
    gw_graph      = build_gateway_graph(get_gateways())
    aviation_graph = build_aviation_graph(generate_aviation_nodes(30))
    upf_graph     = build_upf_graph(get_upfs())
    static_ground = build_static_ground_graph(gw_graph, aviation_graph, upf_graph)

    all_records = []
    history_list = []
    nbr_sims = len(start_times)

    N_AIRCRAFT = 3                      # aircraft per simulation
    START_LAT_MIN, START_LAT_MAX = 30, 50           # degrees
    START_LON_MIN, START_LON_MAX = -50, -20
    END_LAT_MIN, END_LAT_MAX = 30, 50
    END_LON_MIN, END_LON_MAX = 10, 20

    rng = np.random.default_rng(seed=42)
    n = N_AIRCRAFT * nbr_sims
    start_lats = rng.uniform(START_LAT_MIN, START_LAT_MAX, n)
    start_lons = rng.uniform(START_LON_MIN, START_LON_MAX, n)
    end_lats   = rng.uniform(END_LAT_MIN, END_LAT_MAX, n)
    end_lons   = rng.uniform(END_LON_MIN, END_LON_MAX, n)
    for gidx in range(nbr_sims):
        # --- Initialise Satellite Constellation ---
        sats = WalkerDelta(T, P, F, INC, SATELLITE_ALTITUDE).generate()

        # --- Initialise Aircraft ---
        aircrafts = []
        for i in range(N_AIRCRAFT):
            k = N_AIRCRAFT * gidx + i   # simulation gidx uses entries k = 3*gidx, 3*gidx+1, 3*gidx+2
            aircrafts.append(Aircraft((start_lats[k], start_lons[k]),
                                    (end_lats[k], end_lons[k]),
                                    900, f'AC{i+1}'))

        # --- Initialise Dynamic Network ---
        current_time = start_times[gidx]
        network = DynamicNetwork(sats, P, math.floor(T / P), F, current_time, static_ground, aircrafts, seed=seed[gidx])
        mean_delay_by_type = network.queues.mean_delay_by_type()

        # Simulation parameters
        t = 0.0
        t_end = SIM_DURATION
        dt = TIME_STEP

        # Test node-based delay
        if analyse:
            test_queue_delay_dynamics(network, dt=TIME_STEP)

        euro_graph = network.euro_graph
        euro_graph_list = []
        time_list = [current_time]
        overlay_records = []
        history = {}

        while t < t_end:
            tstep = t//dt
            if plot:
                euro_graph_list.append({'t': t, 'graph': euro_graph.copy()})

            data = snapshot_to_pyg(euro_graph)
            history[tstep] = {}
            history[tstep]['t'] = t
            history[tstep]['x'] = data.x
            history[tstep]['edge_index'] = data.edge_index
            history[tstep]['edge_attr'] = data.edge_attr
            history[tstep]['node_names'] = data.node_names
            history[tstep]['name_to_idx'] = data.name_to_idx

            ovls = []
            ovls += network.get_ovls('AC1', 'UPF_DE_Frankfurt',3, current_time)
            ovls += network.get_ovls('AC2', 'UPF_DE_Frankfurt', 3, current_time) 
            ovls += network.get_ovls('AC3', 'UPF_DE_Frankfurt', 3, current_time)  
            print(f'Simulation {gidx} -- Time {t}')
            overlay_records = []
            for ovl in ovls:
                udl = network.get_udl_path(ovl['ovl_path'])
                if udl:
                    delay_ms = network.get_udl_path_delay(udl, current_time)*1000
                    prior_delay_ms = get_udl_prior_delay(network.graph, udl, mean_delay_by_type) * 1000
                    overlay_records.append({'t': t, 'ovl_id': ovl['ovl_id'], 'ovl_path': ovl['ovl_path'],
                                            'udl': udl, 'delay_ms': delay_ms, 'prior_delay_ms': prior_delay_ms})
                    #print(f'udl {udl} -- {network.get_udl_path_delay(udl, current_time)*1000} ms delay at time {t}')
            all_records += overlay_records
            history[tstep]['ovls'] = overlay_records
            (_, euro_graph) = network.update(dt)
            current_time += TimeDelta(dt, format="sec")
            time_list.append(current_time)
            t += dt
        history_list.append(history)
        
        # --- plotting ---
        if plot:
            plot_full_graph_timeline([euro_graph_list[i]['graph'] for i in range(len(euro_graph_list))], time_list, title=f"European Constellation Over Time -- Sim {gidx}")
            
    print(f'Saving data to path: {dataset_path}')
    torch.save({'history_list': history_list}, dataset_path)
    print("Simulation complete. Generating visualization...")
    
    if analyse:
        analyse_overlay_delays([{**r, 'overlay': r['ovl_path']} for r in all_records], dt)
        