# simulation/propagator.py
class Propagator:
    def __init__(self, satellites, aircrafts):
        self.satellites = satellites
        self.aircrafts = aircrafts

    def step(self, dt):
        for sat in self.satellites:
            sat.propagate(dt)

        for ac in self.aircrafts:
            ac.propagate(dt)