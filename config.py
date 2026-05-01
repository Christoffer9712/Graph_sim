# config.py
from astropy import units as u

T = 240              # total satellites
P = 24               # planes
F = 1               # phasing (MUST be 1 for current code to work!)
INC = 53 * u.deg    # inclination
ALTITUDE = 550 * u.km

TIME_STEP = 30 * u.s
SIM_DURATION = 200 * u.s

MAX_LINK_DISTANCE = 20000 * u.km