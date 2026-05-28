import numpy as np
import geopandas as gpd

from .nodes import GroundNode, GroundNodeType
from utils.utils import load_europe_land_shape


def _grid_land_points(land_shape, spacing_deg: float) -> list[tuple[float, float]]:
    """
    Return (lat, lon) pairs for all grid points that fall within land_shape.

    Uses a geopandas spatial join instead of looping over individual points,
    which is orders of magnitude faster for large grids.
    """
    minx, miny, maxx, maxy = land_shape.bounds

    lons = np.arange(minx, maxx + spacing_deg, spacing_deg)
    lats = np.arange(miny, maxy + spacing_deg, spacing_deg)
    grid_lons, grid_lats = np.meshgrid(lons, lats)

    points_gdf = gpd.GeoDataFrame(
        {'lat': grid_lats.ravel(), 'lon': grid_lons.ravel()},
        geometry=gpd.points_from_xy(grid_lons.ravel(), grid_lats.ravel()),
        crs='EPSG:4326',
    )
    land_gdf = gpd.GeoDataFrame(geometry=[land_shape], crs='EPSG:4326')
    within   = gpd.sjoin(points_gdf, land_gdf, how='inner', predicate='within')
    within   = within.sort_values(['lat', 'lon'])   # stable ordering across runs

    return list(zip(within['lat'].values, within['lon'].values))


def generate_aviation_nodes(n_target: int = 300) -> list[GroundNode]:
    """
    Generate n_target aviation nodes on a regular lat/lon grid over
    mainland Europe, excluding any point that falls in the sea.

    Algorithm
    ---------
    1. Binary-search for the largest grid spacing that still yields
       >= n_target land points.  Larger spacing = fewer points, so we
       converge on the coarsest grid that meets the requirement.
    2. Uniformly subsample the resulting list to exactly n_target nodes.
    """
    land = load_europe_land_shape()

    # Binary search: lo = fine (many points), hi = coarse (few points).
    # We want the largest spacing where len(pts) >= n_target.
    lo, hi = 0.2, 5.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        pts = _grid_land_points(land, mid)
        if len(pts) >= n_target:
            lo = mid    # spacing can be even coarser
        else:
            hi = mid    # spacing is too coarse — refine

    pts = _grid_land_points(land, lo)

    # Uniform subsample to exactly n_target
    indices  = np.round(np.linspace(0, len(pts) - 1, n_target)).astype(int)
    selected = [pts[i] for i in indices]

    return [
        GroundNode(
            node_id=f"AV_{i:03d}",
            node_type=GroundNodeType.AVIATION,
            lat=lat, lon=lon,
        )
        for i, (lat, lon) in enumerate(selected)
    ]