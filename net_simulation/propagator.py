"""
Moves satellites and aircraft, and keeps each satellite's geodetic
position (lat, lon, alt) and velocity (dlat, dlon) up to date.
"""
import numpy as np
import astropy.units as unit

from net_simulation.geometry import eci_to_lla_rates


class Propagator:
    def __init__(self, satellites, aircrafts, start_time):
        self.satellites = satellites
        self.aircrafts = aircrafts
        self._update_geodetic(start_time)          # positions valid at t = 0

    def step(self, dt, time):
        """Advance everything by dt seconds; `time` is the new simulation time."""
        for sat in self.satellites:
            sat.propagate(dt)

        for ac in self.aircrafts:
            ac.propagate(dt)

        self._update_geodetic(time)

    def _update_geodetic(self, time):
        """GCRS -> (lat, lon, alt) and (dlat, dlon) for all satellites, in one batched call."""
        r = np.stack([s.position_gcrs.to(unit.m).value for s in self.satellites]) * unit.m
        v = np.stack([s.velocity_gcrs.to(unit.m / unit.s).value for s in self.satellites]) * (unit.m / unit.s)
        lat, lon, alt, dlat, dlon, _ = eci_to_lla_rates(r, v, time)
        for i, sat in enumerate(self.satellites):
            sat.position = (float(lat[i]), float(lon[i]), float(alt[i]))
            sat.velocity = (float(dlat[i]), float(dlon[i]))