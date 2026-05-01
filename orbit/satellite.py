# orbit/satellite.py

from poliastro.twobody import Orbit
from poliastro.bodies import Earth

class Satellite:
    def __init__(self, plane, index, orbit: Orbit):
        self.index = index
        self.plane = plane
        self.orbit = orbit

    def propagate(self, dt):
        self.orbit = self.orbit.propagate(dt)

    @property
    def position(self):
        return self.orbit.r  # (x, y, z)

    @property
    def velocity(self):
        return self.orbit.v