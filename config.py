# config.py
from astropy import units as u

T = 1296              # total satellites
P = 72               # planes
#T = 100               # total satellites
#P = 20                # planes

F = 45               # phasing parameter; adjacent-plane phase = 360 * F / T degrees
INC = 53 * u.deg    # inclination
ALTITUDE = 550 * u.km

TIME_STEP = 30 * u.s
SIM_DURATION = 600 * u.s

MAX_LINK_DISTANCE = 1000 * u.km

MAX_SAT_PER_GW = 5  # Limit the number of satellites each GW connects to

LINK_PER = {
    'gw_sat':            1e-4,
    'a2s':               1e-4,
    'a2g':               1e-3,
    'ground_grid':       1e-9,
}