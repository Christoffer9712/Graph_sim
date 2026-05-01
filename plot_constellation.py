import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_constellation():
    # Points
    data = pd.DataFrame({
        'city': ['London', 'Frankfurt'],
        'latitude': [51.5074, 50.1109],
        'longitude': [-0.1278, 8.6821]
    })

    # Create base map with points
    fig = px.scatter_geo(
        data,
        lat='latitude',
        lon='longitude',
        text='city'
    )

    # Add line between the two cities
    fig.add_trace(go.Scattergeo(
        lat=[51.5074, 50.1109],
        lon=[-0.1278, 8.6821],
        mode='lines',
        line=dict(width=2)
    ))

    # Optional: zoom into Europe
    fig.update_geos(
        scope='europe',
        center=dict(lat=51, lon=5),
        projection_scale=5
    )

    fig.update_layout(title='London to Frankfurt', title_x=0.5)

    fig.show()