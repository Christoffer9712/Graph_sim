# utils/coordinates.py

from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
import astropy.units as u

def eci_to_latlon(r, obstime):
    gcrs = GCRS(
        CartesianRepresentation(r),
        obstime=obstime
    )

    itrs = gcrs.transform_to(ITRS(obstime=obstime))

    lat = itrs.earth_location.lat.to(u.deg)
    lon = itrs.earth_location.lon.to(u.deg)

    return lat.value, lon.value