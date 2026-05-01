# config.py
from astropy import units as u

T = 1296              # total satellites
P = 72               # planes
#T = 24               # total satellites
#P = 6                # planes

F = 45               # phasing parameter; adjacent-plane phase = 360 * F / T degrees
INC = 53 * u.deg    # inclination
ALTITUDE = 550 * u.km

TIME_STEP = 30 * u.s
SIM_DURATION = 3000 * u.s

MAX_LINK_DISTANCE = 20000 * u.km