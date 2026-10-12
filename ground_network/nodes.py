from dataclasses import dataclass
@dataclass(frozen=True)
class GroundNode:
    node_id:   str
    node_type: str
    lat:       float
    lon:       float
    dn:        list[str] | None = None  # Only for UPFs, indicating the data network they belong to (e.g., Internet)