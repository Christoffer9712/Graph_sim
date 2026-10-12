# orbit/satellite.py

from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
class Satellite:
    def __init__(self, plane, index, orbit: Orbit):
        self.index = index
        self.plane = plane
        self.orbit = orbit

    def propagate(self, dt):
        self.orbit = self.orbit.propagate(dt*u.s)

    @property
    def position(self):
        return self.orbit.r  # (x, y, z)

    @property
    def velocity(self):
        return self.orbit.v