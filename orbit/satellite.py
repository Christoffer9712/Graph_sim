# orbit/satellite.py

from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
class Satellite:
    def __init__(self, plane, index, orbit: Orbit, position=(0,0,0), velocity=(0,0)):
        self.index = index
        self.plane = plane
        self.orbit = orbit
        self.position = position
        self.velocity = velocity


    def propagate(self, dt):
        self.orbit = self.orbit.propagate(dt*u.s)

    @property
    def position_gcrs(self):
        return self.orbit.r  # (x, y, z)

    @property
    def velocity_gcrs(self):
        return self.orbit.v