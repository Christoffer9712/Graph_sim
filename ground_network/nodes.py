from dataclasses import dataclass
from enum import IntEnum


class GroundNodeType(IntEnum):
    GATEWAY  = 1   # Type 1 — satellite gateway
    AVIATION = 2   # Type 2 — aviation network node
    UPF      = 3   # Type 3 — user plane function


@dataclass(frozen=True)
class GroundNode:
    node_id:   str
    node_type: GroundNodeType
    lat:       float
    lon:       float