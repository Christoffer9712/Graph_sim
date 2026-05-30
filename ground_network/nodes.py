from dataclasses import dataclass
from enum import IntEnum


class GroundNodeType(str):
    GATEWAY  = 'gateway'   # Type 1 — satellite gateway
    AVIATION = 'aviation'   # Type 2 — aviation network node
    UPF      = 'upf'   # Type 3 — user plane function


@dataclass(frozen=True)
class GroundNode:
    node_id:   str
    node_type: str
    lat:       float
    lon:       float
    dn:        list[str] | None = None  # Only for UPFs, indicating the data network they belong to (e.g., Internet)