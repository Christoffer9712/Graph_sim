# config.py
from dataclasses import dataclass
from enum import Enum
from astropy import units as u

# -------- Orbit and constellation parameters --------
T = 1296              # total satellites
P = 72               # planes
#T = 100               # total satellites
#P = 20                # planes

F = 45               # phasing parameter; adjacent-plane phase = 360 * F / T degrees
INC = 53 * u.deg    # inclination
SATELLITE_ALTITUDE = 550_000 * u.m

# -------- Aircraft parameters --------
AIRCRRAFT_ALTITUDE = 10_000

# -------- Simulation parameters --------
TIME_STEP = 30 * u.s
SIM_DURATION = 100
TUNNELING_DECISION_INTERVAL = 100
ALWAYS_UP = True  # If True, links never go DOWN (for testing visualization without stochasticity)

# -------- Link parameters --------
MAX_ISL_LINK_DISTANCE = 1000_000.0  # Maximum distance for inter-satellite links (1000 km)
MAX_A2G_LINK_DISTANCE = 200_000.0  # Maximum slant range for aircraft-to-ground-node links (200 km)
MAX_GROUND_LINK_DISTANCE = 1000_000.0  # Maximum distance for direct ground node links (1000 km)

MAX_SAT_PER_GW = 5  # Limit the number of satellites each GW connects to
MAX_SAT_PER_AC = 3  # Limit the number of satellites each aircraft connects to
MAX_GROUND_NEIGHBORS = 4  # Limit the number of direct ground links per node

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
    mttf_ref:   float
    d_ref:      float
    mttr:       float
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
    LinkType.SA2A: LinkParams(
        mttf_ref=60, d_ref=1000, mttr=50,  gamma_fail=1.5,
        per_ref=1e-3,    gamma_per=2.0, alpha_load=0.5, cv=0.4,
    ),
    LinkType.DA2G: LinkParams(
        mttf_ref=60, d_ref=100, mttr=50,  gamma_fail=1.0,
        per_ref=1e-3,    gamma_per=2.0, alpha_load=0.5, cv=0.1,
    ),
    LinkType.FEEDER_LINK: LinkParams(
        mttf_ref=1000, d_ref=1000, mttr=50,  gamma_fail=1.0,
        per_ref=1e-4,    gamma_per=2.0, alpha_load=0.5, cv=0.3,
    ),
    LinkType.GROUND_GRID: LinkParams( # Assume ground network is static for now with a fixed PER
        mttf_ref=0, d_ref=0, mttr=0,  gamma_fail=0,
        per_ref=1e-8,    gamma_per=0, alpha_load=0, cv=0,
    ),
}

# -------- Physical constants --------
R_EARTH_M  = 6_371_000.0
C_VACUUM   = 3e8     # m/s  — wireless / space links
C_FIBER    = 2e8     # m/s  — ~0.67c in glass fibre