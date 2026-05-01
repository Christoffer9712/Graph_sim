# config.py
from astropy import units as u

T = 1296              # total satellites
P = 72               # planes
T = 400               # total satellites
P = 20                # planes

F = 45               # phasing parameter; adjacent-plane phase = 360 * F / T degrees
INC = 53 * u.deg    # inclination
ALTITUDE = 550 * u.km

TIME_STEP = 30 * u.s
SIM_DURATION = 100 * u.s

MAX_LINK_DISTANCE = 1000 * u.km