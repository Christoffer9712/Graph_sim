"""
Pure geometry helpers: no graph, no state.

Positions are geodetic tuples (lat [deg], lon [deg], alt [m]) on WGS84.
"""
import numpy as np
import astropy.units as unit
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, CartesianDifferential

WGS84_A = 6378137.0           # semi-major axis [m]
WGS84_E2 = 6.69437999014e-3    # first eccentricity squared


def lla_to_ecef_m(lat_deg, lon_deg, alt_m):
    """Geodetic (WGS84) -> ECEF [m]. Works on scalars or arrays."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl**2)
    return np.stack([(N + alt_m) * cl * np.cos(lon),
                     (N + alt_m) * cl * np.sin(lon),
                     (N * (1.0 - WGS84_E2) + alt_m) * sl], axis=-1)


def dist_m(a_pos, b_pos) -> float:
    """Straight-line distance [m] between two (lat [deg], lon [deg], alt [m]) positions."""
    return float(np.linalg.norm(lla_to_ecef_m(*a_pos) - lla_to_ecef_m(*b_pos)))


def eci_to_lla_rates(positions, velocities, time):
    """GCRS position/velocity -> geodetic lat, lon [deg], alt [m] and their
    Earth-fixed rates dlat/dt, dlon/dt [deg/s], dalt/dt [m/s].

    positions, velocities : Quantity of shape (3,) or (N, 3), e.g. orbit.r, orbit.v
    time                  : astropy Time at which the state is valid
    """
    rep = CartesianRepresentation(positions.T,
                                  differentials=CartesianDifferential(velocities.T))
    itrs = GCRS(rep, obstime=time).transform_to(ITRS(obstime=time))
    v = itrs.velocity.d_xyz.to(unit.m / unit.s).value      # ECEF velocity, Earth rotation included

    loc = itrs.earth_location
    lat, lon, h = loc.lat.rad, loc.lon.rad, loc.height.to(unit.m).value

    # Velocity in local east / north / up
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    v_east = -so * v[0] + co * v[1]
    v_north = -sl * co * v[0] - sl * so * v[1] + cl * v[2]
    v_up = cl * co * v[0] + cl * so * v[1] + sl * v[2]

    # Ellipsoid radii of curvature
    w = 1.0 - WGS84_E2 * sl**2
    N = WGS84_A / np.sqrt(w)                        # prime vertical (east-west)
    M = WGS84_A * (1.0 - WGS84_E2) / w**1.5         # meridian (north-south)

    dlat = np.degrees(v_north / (M + h))
    dlon = np.degrees(v_east / ((N + h) * cl))
    return np.degrees(lat), np.degrees(lon), h, dlat, dlon, v_up


def slant_range_m(sat_alt_m, elevation_deg, re_m=6_371_000.0):
    el = np.radians(elevation_deg)
    r = re_m + sat_alt_m
    return np.sqrt(r**2 - (re_m * np.cos(el))**2) - re_m * np.sin(el)


def is_in_europe(lat, lon):
    # Rough bounding box for Europe
    return 20 <= lat <= 70 and -60 <= lon <= 40  # Includes North Atlantic