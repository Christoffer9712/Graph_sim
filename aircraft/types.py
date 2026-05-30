from dataclasses import dataclass
from enum import IntEnum
from config import LinkType


@dataclass(frozen=True)
class TrafficDescription:
    fiveQI: int
    BW:     float   # Mbps
    DN:    str

@dataclass(frozen=True)
class TunnelDescription:
    fiveQI:   int
    BW:       float
    linkType: LinkType
    firstHop: str
    GW:       str    # empty string for DA2G tunnels
    UPF:      str