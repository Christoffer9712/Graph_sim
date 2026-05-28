import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from astropy import units as unit
from config import LinkType
from ground_network.nodes import GroundNodeType
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


def _full_graph_edge_traces(graph, node_coords):
    categories = {
        'satellite_links': {
            'lat': [], 'lon': [], 'color': 'royalblue', 'name': 'sat-sat',
        },
        'feeder_links': {
            'lat': [], 'lon': [], 'color': 'orange', 'name': 'satellite-gateway',
        },
        'ground_links': {
            'lat': [], 'lon': [], 'color': 'gray', 'name': 'ground-grid',
        },
    }

    for u, v, data in graph.edges(data=True):
        if u not in node_coords or v not in node_coords:
            continue
        if data.get('state') and data.get('state') != 'UP':
            continue

        if data['link_type'] == LinkType.INTRA_PLANE_ISL or data['link_type'] == LinkType.INTER_PLANE_ISL:
            category = 'satellite_links'    
        elif data['link_type'] == LinkType.FEEDER_LINK:
            category = 'feeder_links'
        else:
            category = 'ground_links'

        lat_u, lon_u = node_coords[u]
        lat_v, lon_v = node_coords[v]
        categories[category]['lat'] += [lat_u, lat_v, None]
        categories[category]['lon'] += [lon_u, lon_v, None]

    traces = []
    for category in categories.values():
        if category['lat']:
            traces.append(go.Scattergeo(
                lat=category['lat'],
                lon=category['lon'],
                mode='lines',
                line=dict(width=1, color=category['color']),
                showlegend=False,
                hoverinfo='skip',
            ))
    return traces


def plot_full_graph_timeline(full_graph_list, time_list, title="Full Graph Timeline"):
    """Plot the full graph sequence as an animated timeline.

    This includes all satellite and ground nodes, as well as every edge
    present in each full-graph snapshot. Ground node positions are assumed
    to be static and are plotted at their fixed latitude/longitude.
    """
    rows = []
    coord_cache = {}

    for t_idx, (graph, current_time) in enumerate(zip(full_graph_list, time_list)):
        coord_cache[t_idx] = {}
        aircraft_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'aircraft']
        satellite_nodes = [n for n, d in graph.nodes(data=True) if 'position' in d and d.get('node_type') != 'aircraft']
        ground_nodes = [n for n, d in graph.nodes(data=True) if 'lat' in d and 'lon' in d]

        if satellite_nodes:
            positions = np.stack([graph.nodes[node]['position'] for node in satellite_nodes])
            latitudes, longitudes = eci_to_latlon_batch(positions, current_time)
            for idx, node in enumerate(satellite_nodes):
                coord_cache[t_idx][node] = (latitudes[idx], longitudes[idx])
                rows.append({
                    "time": t_idx,
                    "node": node,
                    "latitude": latitudes[idx],
                    "longitude": longitudes[idx],
                    "type": "satellite",
                })

        for node in ground_nodes:
            lat, lon = graph.nodes[node]['lat'], graph.nodes[node]['lon']
            coord_cache[t_idx][node] = (lat, lon)
            rows.append({
                "time": t_idx,
                "node": node,
                "latitude": lat,
                "longitude": lon,
                "type": "ground",
            })

        for node in aircraft_nodes:
            lat, lon = graph.nodes[node]['lat'], graph.nodes[node]['lon']
            coord_cache[t_idx][node] = (lat, lon)
            rows.append({
                "time": t_idx,
                "node": node,
                "latitude": lat,
                "longitude": lon,
                "type": "aircraft",
            })

    dataframe = pd.DataFrame(rows)

    fig = px.scatter_geo(
        dataframe,
        lat="latitude",
        lon="longitude",
        animation_frame="time",
        animation_group="node",
        color="type",
        symbol="type",
        text="node",
        title=f"{title} (ground nodes fixed, satellite nodes move)",
    )

    for _ in range(4):
        fig.add_trace(go.Scattergeo(lat=[], lon=[], mode='lines', showlegend=False, hoverinfo='skip'))

    for t_idx, graph in enumerate(full_graph_list):
        edge_traces = _full_graph_edge_traces(graph, coord_cache[t_idx])
        fig.frames[t_idx].data = tuple(fig.frames[t_idx].data) + tuple(edge_traces)

    fig.update_layout(title_x=0.5)
    fig.show()