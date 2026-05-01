import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.coordinates import eci_to_latlon

def _edge_trace(graph, current_time):
    """All UP edges packed into a single Scattergeo trace."""
    lat_all, lon_all = [], []
    for u, v in graph.edges:
        if graph.edges[u, v]['state'] == 'UP':
            lat_u, lon_u = eci_to_latlon(graph.nodes[u]['position'], current_time)
            lat_v, lon_v = eci_to_latlon(graph.nodes[v]['position'], current_time)
            lat_all += [lat_u.item(), lat_v.item(), None]
            lon_all += [lon_u.item(), lon_v.item(), None]

    return go.Scattergeo(
        lat=lat_all,
        lon=lon_all,
        mode='lines',
        line=dict(width=1, color='royalblue'),
        showlegend=False,
        hoverinfo='skip',
    )


def plot_constellation_timeline(graph_list, time_list):
    rows = []
    for t_idx, graph in enumerate(graph_list):
        current_time = time_list[t_idx]
        for node in graph.nodes:
            lat, lon = eci_to_latlon(graph.nodes[node]['position'], current_time)
            rows.append({"time": t_idx, "sat": node,
                         "latitude": lat.item(), "longitude": lon.item()})

    dataframe = pd.DataFrame(rows)

    fig = px.scatter_geo(
        dataframe,
        lat="latitude", lon="longitude",
        animation_frame="time", animation_group="sat",
        text="sat"
    )

    # Register exactly ONE edge placeholder slot in fig.data
    fig.add_trace(go.Scattergeo(lat=[], lon=[], mode='lines',
                                showlegend=False, hoverinfo='skip'))

    # Each frame gets exactly one edge trace appended
    for t_idx, (graph, current_time) in enumerate(zip(graph_list, time_list)):
        fig.frames[t_idx].data = tuple(fig.frames[t_idx].data) + (
            _edge_trace(graph, current_time),
        )

    # Populate the initial view
    #fig.add_trace(_edge_trace(graph_list[0], time_list[0]))

    fig.update_layout(title="Satellite Constellation Over Time", title_x=0.5)
    fig.show()

'''
def _edge_traces(graph, current_time, frame=False):
    """
    Returns a list of Scattergeo traces (one per edge) for a given graph snapshot.
    Set frame=True when building frame data (slightly different object type needed).
    """
    traces = []
    for u, v in graph.edges:
        lat_u, lon_u = eci_to_latlon(graph.nodes[u]['position'], current_time)
        lat_v, lon_v = eci_to_latlon(graph.nodes[v]['position'], current_time)

        if graph.edges[u, v]['state'] == 'UP':
            print(f"Adding edge trace between {u} and {v} at time {current_time.iso}")
            traces.append(go.Scattergeo(
                lat=[lat_u.item(), lat_v.item()],
                lon=[lon_u.item(), lon_v.item()],
                mode='lines',
                line=dict(width=1, color='royalblue'),
                showlegend=False,
                hoverinfo='skip',
            ))
    return traces


def plot_constellation_timeline(graph_list, time_list):
    """
    Builds an animated scatter_geo plot over time, including edges.
    """
    rows = []

    for t_idx, graph in enumerate(graph_list):
        current_time = time_list[t_idx]
        for node in graph.nodes:
            lat, lon = eci_to_latlon(graph.nodes[node]['position'], current_time)
            rows.append({
                "time": t_idx,
                "sat": node,
                "latitude": lat.item(),
                "longitude": lon.item()
            })

    dataframe = pd.DataFrame(rows)

    # --- base animated figure (nodes only) ---
    fig = px.scatter_geo(
        dataframe,
        lat="latitude",
        lon="longitude",
        animation_frame="time",
        animation_group="sat",
        text="sat"
    )

    # --- inject edge traces into each frame ---
    for t_idx, (graph, current_time) in enumerate(zip(graph_list, time_list)):
        edge_traces = _edge_traces(graph, current_time)
        fig.frames[t_idx].data = tuple(fig.frames[t_idx].data) #+ tuple(edge_traces)

    # Add edges for the initial (frame 0) view too
    for trace in _edge_traces(graph_list[0], time_list[0]):
        fig.add_trace(trace)

    fig.update_layout(
        title="Satellite Constellation Over Time",
        title_x=0.5
    )

    fig.show()


def plot_constellation(graph, current_time):
    n, lat, lon = [], [], []

    for node in graph.nodes:
        n.append(f'{graph.nodes[node]['plane']}-{graph.nodes[node]['index']}')
        lat_n, lon_n = eci_to_latlon(graph.nodes[node]['position'], current_time)
        lat.append(lat_n.item())
        lon.append(lon_n.item())

    node_lookup = dict(zip(n, zip(lat, lon)))

    # --- build figure with graph_objects for full control ---
    fig = go.Figure()

    # Edge traces
    for u, v in graph.edges:
        lat_u, lon_u = node_lookup[u]
        lat_v, lon_v = node_lookup[v]
        if graph.edges[u, v]['state'] == 'UP':
            fig.add_trace(go.Scattergeo(
                lat=[lat_u, lat_v],
                lon=[lon_u, lon_v],
                mode='lines',
                line=dict(width=1, color='royalblue'),
                showlegend=False,
                hoverinfo=f'PER: {graph.edges[u, v]["per"]:.2e}, Dist: {graph.edges[u, v]["distance"]:.1f} km',
            ))

    # Node trace
    fig.add_trace(go.Scattergeo(
        lat=lat,
        lon=lon,
        mode='markers+text',
        text=n,
        textposition='top center',
        marker=dict(size=6, color='red'),
        name='Satellites'
    ))

    fig.update_layout(title='Satellite network', title_x=0.5)
    fig.show()


def plot_ground_track(latlons):
    lats, lons = zip(*latlons)
    plt.figure()
    plt.scatter(lons, lats, s=10)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Satellite Ground Track")
    plt.show()
'''