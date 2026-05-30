from .nodes import GroundNode, GroundNodeType

_UPF_LOCATIONS = [
    # (node_id,                  lat,     lon,   description,              DN)
    ("UPF_DE_Frankfurt",       50.110,   8.682, "Frankfurt, Germany (DE-CIX)", ["Internet", "ATM_Frankfurt"]),
    ("UPF_NL_Amsterdam",       52.378,   4.900, "Amsterdam, Netherlands (AMS-IX)", ["Internet", "ATM_Amsterdam"]),
    ("UPF_SE_Stockholm",       59.334,  18.063, "Stockholm, Sweden", ["Internet", "ATM_Stockholm"]),
    ("UPF_PL_Warsaw",          52.230,  21.010, "Warsaw, Poland", ["Internet", "ATM_Warsaw"]),
    ("UPF_IT_Milan",           45.465,   9.186, "Milan, Italy", ["Internet", "ATM_Milan"]),
    ("UPF_GB_London",          51.507,  -0.128, "London, United Kingdom (LINX)", ["Internet", "ATM_London"]),
    ("UPF_FR_Paris",           48.857,   2.352, "Paris, France", ["Internet", "ATM_Paris"]),
    ("UPF_ES_Madrid",          40.417,  -3.704, "Madrid, Spain", ["Internet", "ATM_Madrid"]),
    ("UPF_CH_Zurich",          47.377,   8.542, "Zurich, Switzerland", ["Internet", "ATM_Zurich"]),
    ("UPF_FI_Helsinki",        60.170,  24.938, "Helsinki, Finland", ["Internet", "ATM_Helsinki"]),
]

def get_upfs() -> list[GroundNode]:
    """Return GroundNode objects for all UPF locations."""
    return [
        GroundNode(node_id=nid, node_type=GroundNodeType.UPF, lat=lat, lon=lon, dn=dn)
        for nid, lat, lon, _, dn in _UPF_LOCATIONS
    ]