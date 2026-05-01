from .nodes import GroundNode, GroundNodeType

_GW_LOCATIONS = [
    # (node_id,               lat,     lon,    description)
    ("GW_NL_Burum",         53.357,   6.198, "Burum, Netherlands"),
    ("GW_IT_SestoCalende",  45.730,   8.630, "Sesto Calende, Italy"),
    ("GW_SE_Stockholm",     59.440,  18.070, "Stockholm, Sweden"),
    ("GW_SE_Lulea",         65.580,  22.150, "Luleå, Sweden"),
    ("GW_FI_Helsinki",      60.170,  24.940, "Helsinki, Finland"),
    ("GW_PL_Gdynia",        54.520,  18.530, "Gdynia, Poland"),
    ("GW_PL_Poznan",        52.400,  16.900, "Poznan, Poland"),
    ("GW_DE_Berlin",        52.520,  13.405, "Berlin, Germany"),
    ("GW_DE_Munich",        48.100,  11.600, "Munich, Germany"),
    ("GW_CZ_Prague",        50.100,  14.400, "Prague, Czech Republic"),
    ("GW_FR_Toulouse",      43.600,   1.440, "Toulouse, France"),
    ("GW_ES_Madrid",        40.500,  -3.700, "Madrid, Spain"),
    ("GW_PT_Aveiro",        40.640,  -8.650, "Aveiro, Portugal"),
    ("GW_IT_Rome",          41.900,  12.500, "Rome, Italy"),
]


def get_gateways() -> list[GroundNode]:
    """Return GroundNode objects for all known European Starlink GWs."""
    return [
        GroundNode(node_id=nid, node_type=GroundNodeType.GATEWAY, lat=lat, lon=lon)
        for nid, lat, lon, _ in _GW_LOCATIONS
    ]