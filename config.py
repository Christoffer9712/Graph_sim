# config.py
from dataclasses import dataclass
from enum import Enum

from astropy import units as u

T = 1296              # total satellites
P = 72               # planes
#T = 100               # total satellites
#P = 20                # planes

F = 45               # phasing parameter; adjacent-plane phase = 360 * F / T degrees
INC = 53 * u.deg    # inclination
ALTITUDE = 550_000 * u.m

TIME_STEP = 30 * u.s
SIM_DURATION = 80 * u.s

MAX_LINK_DISTANCE = 1000_000 * u.m

MAX_SAT_PER_GW = 5  # Limit the number of satellites each GW connects to

LINK_PER = {
    'gw_sat':            1e-4,
    'a2s':               1e-4,
    'a2g':               1e-3,
    'ground_grid':       1e-9,
}

# ──────────────────────────────────────────────────────────────
# Link parameter tables  (Tables 1 & 2)
# All stored values are plain floats: distances in km, times in minutes.
# ──────────────────────────────────────────────────────────────

ALWAYS_UP = True  # If True, links never go DOWN (for testing visualization without stochasticity)

class LinkType(Enum):
    INTRA_PLANE_ISL = "intra_plane_isl"
    INTER_PLANE_ISL = "inter_plane_isl"
    SA2A = "sa2a"
    DA2G = "da2g"
    FEEDER_LINK = "feeder_link"
    GROUND_GRID = "ground_grid"

@dataclass(frozen=True)
class LinkParams:
    # --- Availability (Table 1) ---
    mttf_ref:   float   # minutes
    d_ref:      float   # km
    mttr:       float   # minutes
    gamma_fail: float
    # --- PER (Table 2) ---
    per_ref:    float
    gamma_per:  float
    alpha_load: float
    cv:         float

# Dimensions are s and km (values are not realistic, just for testing)
LINK_PARAMS: dict[LinkType, LinkParams] = {
    LinkType.INTRA_PLANE_ISL: LinkParams(
        mttf_ref=1_000, d_ref=2400, mttr=50,  gamma_fail=0.5,
        per_ref=1e-5,    gamma_per=2.0, alpha_load=0.5, cv=0.1,
    ),
    LinkType.INTER_PLANE_ISL: LinkParams(
        mttf_ref=3_000,  d_ref=1000, mttr=1000, gamma_fail=1.0,
        per_ref=1e-4,    gamma_per=2.0, alpha_load=0.5, cv=0.2,
    ),
# For now, link model for SA2A, DA2G, Feeder Link, and ground links are simpler
#    LinkType.SA2A: LinkParams(
#        mttf_ref=60, d_ref=1000, mttr=50,  gamma_fail=1.5,
#        per_ref=1e-3,    gamma_per=2.0, alpha_load=0.5, cv=0.4,
#    ),
#    LinkType.DA2G: LinkParams(
#        mttf_ref=60, d_ref=100, mttr=50,  gamma_fail=1.0,
#        per_ref=1e-3,    gamma_per=2.0, alpha_load=0.5, cv=0.1,
#    ),
    LinkType.FEEDER_LINK: LinkParams(
        mttf_ref=1000, d_ref=1000, mttr=50,  gamma_fail=1.0,
        per_ref=1e-4,    gamma_per=2.0, alpha_load=0.5, cv=0.3,
    ),
    LinkType.GROUND_GRID: LinkParams( # Assume ground network is static for now with a fixed PER
        mttf_ref=0, d_ref=0, mttr=0,  gamma_fail=0,
        per_ref=1e-8,    gamma_per=0, alpha_load=0, cv=0,
    ),
}

R_EARTH_M  = 6_371_000.0
C_VACUUM   = 3e8     # m/s  — wireless / space links
C_FIBER    = 2e8     # m/s  — ~0.67c in glass fibre

# Maximum slant range for aircraft-to-ground-node links (200 km).
A2G_RANGE_M = 200_000.0