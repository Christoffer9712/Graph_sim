from config import FIVEQI_TABLE

def cost_function(fiveQI: int, bw: float, per: float, latency: float) -> float:
    per_target, latency_target = FIVEQI_TABLE[fiveQI]
    
    per_cost = 0.0
    latency_cost = 0.0
    if per > per_target:
        per_cost = (1+(per - per_target) / per_target)**2  # Quadratic penalty for PER above target

    if latency > latency_target:
        latency_cost = (1+(latency - latency_target) / latency_target)**2  # Quadratic penalty for latency above target

    print(f"Cost function for fiveQI={fiveQI}, bw={bw}Mbps, per={per:.2e}, latency={latency:.3f}s, per_target={per_target:.2e}, latency_target={latency_target:.3f}s: per_cost={per_cost:.2f}, latency_cost={latency_cost:.2f}, total_cost={(bw/1000_000) * (per_cost + latency_cost):.2f}")
    return (bw/1000_000) * (per_cost + latency_cost)