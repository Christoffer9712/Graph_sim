import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from astropy import units as unit
from utils.coordinates import eci_to_latlon, eci_to_latlon_batch


def _edge_trace(graph, node_coords):
    """
    Builds a single Scattergeo trace for all UP edges using pre-computed
    node coordinates, avoiding any calls to eci_to_latlon.

    node_coords: dict mapping node_id -> (lat, lon) for this time step.
    """
    lat_all, lon_all = [], []
    for u, v in graph.edges:
        if graph.edges[u, v]['state'] == 'UP':
            lat_u, lon_u = node_coords[u]
            lat_v, lon_v = node_coords[v]
            lat_all += [lat_u, lat_v, None]
            lon_all += [lon_u, lon_v, None]

    return go.Scattergeo(
        lat=lat_all,
        lon=lon_all,
        mode='lines',
        line=dict(width=1, color='royalblue'),
        showlegend=False,
        hoverinfo='skip',
    )


def plot_constellation_timeline(graph_list, time_list, title="Satellite Constellation Over Time"):
    rows = []

    # coord_cache[t_idx][node] = (lat, lon)
    # Built here so eci_to_latlon is called exactly once per (node, time) pair.
    coord_cache = {}

    for t_idx, (graph, current_time) in enumerate(zip(graph_list, time_list)):
        coord_cache[t_idx] = {}

        positions = np.stack([graph.nodes[node]['position'] for node in graph.nodes])
        latitudes, longitudes = eci_to_latlon_batch(positions, current_time)
        for idx, node in enumerate(graph.nodes):
            coord_cache[t_idx][node] = (latitudes[idx], longitudes[idx])
            rows.append({"time": t_idx, "sat": node,
                         "latitude": latitudes[idx], "longitude": longitudes[idx]})

    dataframe = pd.DataFrame(rows)

    fig = px.scatter_geo(
        dataframe,
        lat="latitude", lon="longitude",
        animation_frame="time", animation_group="sat",
        text="sat"
    )

    # One permanent empty slot in fig.data that every frame will update in-place.
    fig.add_trace(go.Scattergeo(lat=[], lon=[], mode='lines',
                                showlegend=False, hoverinfo='skip'))

    for t_idx, graph in enumerate(graph_list):
        edge_trace = _edge_trace(graph, coord_cache[t_idx])
        fig.frames[t_idx].data = tuple(fig.frames[t_idx].data) + (edge_trace,)

    # Populate the edge slot for the initial static view (before play is pressed).
    #fig.data[-1] = _edge_trace(graph_list[0], coord_cache[0])

    fig.update_layout(title=title, title_x=0.5)
    fig.show()