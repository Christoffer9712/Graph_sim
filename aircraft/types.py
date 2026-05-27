from dataclasses import dataclass
from enum import IntEnum


@dataclass(frozen=True)
class TrafficDescription:
    fiveQI: int
    BW:     float   # Mbps
    UPF:    str


class LinkType(IntEnum):
    SA2A = 1   # satellite air-to-air link
    DA2G = 2   # direct air-to-ground link


@dataclass(frozen=True)
class TunnelDescription:
    fiveQI:   int
    BW:       float
    linkType: LinkType
    firstHop: str
    GW:       str    # empty string for DA2G tunnels
    UPF:      str