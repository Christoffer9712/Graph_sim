import numpy as np
from functools import lru_cache
from astropy import units as u
from shapely.geometry import box

R_EARTH = 6_371.0  # earth radius in km

EUROPE_BBOX = box(
    -25,   # west (Azores-ish)
    34,    # south (Mediterranean)
    45,    # east (Ukraine)
    72     # north (Scandinavia)
)

def latlon_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    """Geodetic lat/lon/alt → ECEF XYZ (metres)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r   = R_EARTH + alt_m
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat),
    ])


def eci_to_altitude(eci_position) -> float:
    """Altitude above Earth's surface (metres) from an ECI position vector."""
    return float(np.linalg.norm(eci_position / u.km)) - R_EARTH


def elevation_angle_deg(
    ground_lat: float, ground_lon: float,
    sat_lat:    float, sat_lon:    float,
    sat_alt_m:  float,
) -> float:
    """
    Elevation angle (degrees) from a ground point to a satellite.
    Positive = above horizon.

    Method: project the ground→satellite vector onto the outward surface
    normal at the ground point; arcsin of that gives the elevation.
    """
    G    = latlon_to_ecef(ground_lat, ground_lon, 0.0)
    S    = latlon_to_ecef(sat_lat,    sat_lon,    sat_alt_m)
    v    = S - G
    n    = G / np.linalg.norm(G)          # unit outward normal at G
    sin_el = np.dot(v, n) / np.linalg.norm(v)
    return float(np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0))))


@lru_cache(maxsize=1)
def load_europe_land_shape():
    """
    Return a single Shapely geometry covering mainland Europe land.

    Cached — the shapefile is read only once per process.
    Requires geopandas. Handles both old (< 0.14) and new geopandas APIs.
    """
    import geopandas as gpd
    from shapely.ops import unary_union

    try:                                         # geopandas < 0.14
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    except AttributeError:                       # geopandas >= 0.14
        world = gpd.read_file(
            "https://naturalearth.s3.amazonaws.com/50m_cultural/"
            "ne_50m_admin_0_countries.zip"
        )

    # Naturalearth name strings — excludes UK, Ireland, Iceland, island territories
    MAINLAND_EUROPE = {
        'Albania', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herz.',
        'Bulgaria', 'Croatia', 'Czech Rep.', 'Denmark', 'Estonia',
        'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Italy',
        'Latvia', 'Lithuania', 'Luxembourg', 'Macedonia', 'Moldova',
        'Montenegro', 'Netherlands', 'Norway', 'Poland', 'Portugal',
        'Romania', 'Serbia', 'Slovakia', 'Slovenia', 'Spain',
        'Sweden', 'Switzerland', 'Ukraine',
    }

    europe = world[world['NAME'].isin(MAINLAND_EUROPE)]
    europe_clipped = europe.geometry.intersection(EUROPE_BBOX)
    return unary_union(europe_clipped)