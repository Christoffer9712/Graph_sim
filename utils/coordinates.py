# utils/coordinates.py

from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
import astropy.units as u
import numpy as np

def eci_to_latlon(r, obstime):
    gcrs = GCRS(
        CartesianRepresentation(r),
        obstime=obstime
    )

    itrs = gcrs.transform_to(ITRS(obstime=obstime))

    lat = itrs.earth_location.lat.to(u.deg)
    lon = itrs.earth_location.lon.to(u.deg)

    return lat.value, lon.value


def eci_to_latlon_batch(positions, obstime):
    """Convert an array of ECI positions to latitude and longitude.

    Parameters
    ----------
    positions : Quantity or array-like, shape (N, 3)
        Satellite position vectors in ECI coordinates.
    obstime : Time
        Observation time for the coordinate transformation.

    Returns
    -------
    lat : ndarray, shape (N,)
    lon : ndarray, shape (N,)
    """
    pos = np.asarray(positions)
    if pos.ndim == 1:
        pos = pos[np.newaxis, :]

    if isinstance(positions, u.Quantity):
        pos_qty = positions
    else:
        pos_qty = pos * u.km

    gcrs = GCRS(
        CartesianRepresentation(pos_qty.T),
        obstime=obstime
    )
    itrs = gcrs.transform_to(ITRS(obstime=obstime))

    lat = itrs.earth_location.lat.to(u.deg)
    lon = itrs.earth_location.lon.to(u.deg)

    return lat.value, lon.value