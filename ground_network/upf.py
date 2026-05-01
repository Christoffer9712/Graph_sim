from .nodes import GroundNode, GroundNodeType

_UPF_LOCATIONS = [
    # (node_id,                  lat,     lon,   description)
    ("UPF_DE_Frankfurt",       50.110,   8.682, "Frankfurt, Germany (DE-CIX)"),
    ("UPF_NL_Amsterdam",       52.378,   4.900, "Amsterdam, Netherlands (AMS-IX)"),
    ("UPF_SE_Stockholm",       59.334,  18.063, "Stockholm, Sweden"),
    ("UPF_PL_Warsaw",          52.230,  21.010, "Warsaw, Poland"),
    ("UPF_IT_Milan",           45.465,   9.186, "Milan, Italy"),
]


def get_upfs() -> list[GroundNode]:
    """Return GroundNode objects for all UPF locations."""
    return [
        GroundNode(node_id=nid, node_type=GroundNodeType.UPF, lat=lat, lon=lon)
        for nid, lat, lon, _ in _UPF_LOCATIONS
    ]