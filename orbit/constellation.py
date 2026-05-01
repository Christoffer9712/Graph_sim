# orbit/constellation.py

import numpy as np
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
from .satellite import Satellite

class WalkerDelta:
    def __init__(self, T, P, F, INC, altitude):
        self.inc = INC
        self.T = T
        self.P = P
        self.F = F
        self.S = T // P # Satellites per plane
        self.altitude = altitude

    def generate(self):
        satellites = []

        for p in range(self.P):
            raan = 2 * np.pi * p / self.P * u.rad

            for s in range(self.S):
                mean_anomaly = (
                    2 * np.pi * s / self.S +
                    2 * np.pi * self.F * p / self.T
                ) * u.rad

                orbit = Orbit.from_classical(
                    Earth,
                    a=self.altitude + Earth.R,
                    ecc=0 * u.one,
                    inc=self.inc,
                    raan=raan,
                    argp=0 * u.deg,
                    nu=mean_anomaly
                )

                satellites.append(Satellite(p, s, orbit))

        return satellites