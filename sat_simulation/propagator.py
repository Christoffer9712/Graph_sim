# simulation/propagator.py

class Propagator:
    def __init__(self, satellites):
        self.satellites = satellites

    def step(self, dt):
        for sat in self.satellites:
            sat.propagate(dt)