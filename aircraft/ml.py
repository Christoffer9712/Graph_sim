from config import FIVEQI_TABLE, LinkType
from .types import TunnelDescription
def cost_function(fiveQI: int, bw: float, per: float, latency: float) -> float:
    per_target, latency_target = FIVEQI_TABLE[fiveQI]
    
    per_cost = 0.0
    latency_cost = 0.0
    if per > per_target:
        per_cost = (1+(per - per_target) / per_target)

    if latency > latency_target:
        latency_cost = (1+(latency - latency_target) / latency_target)

    print(f"Cost function for fiveQI={fiveQI}, bw={bw}Mbps, per={per:.2e}, latency={latency:.3f}s, per_target={per_target:.2e}, latency_target={latency_target:.3f}s: per_cost={per_cost:.2f}, latency_cost={latency_cost:.2f}, total_cost={(bw/1000_000) * (per_cost + latency_cost):.2f}")
    return (bw/10_000_000) * (per_cost + latency_cost)

def initial_tunnel_cost(prev_tunnels: list[TunnelDescription], current_tunnels: list[TunnelDescription]) -> float:
    init_cost = 0.0
    if len(current_tunnels) == 0:
        return init_cost
    
    if len(prev_tunnels) == 0:
        nbrSa2a = sum([1 for tunnel in current_tunnels if tunnel.linkType == LinkType.SA2A])
        nbrDa2g = sum([1 for tunnel in current_tunnels if tunnel.linkType == LinkType.DA2G])
        init_cost = 100*(nbrDa2g + nbrSa2a)
        print(f'initialised {nbrSa2a} sa2a tunnels and {nbrDa2g} da2g tunnels with cost {init_cost}')
    else:
        for tunnel in prev_tunnels:
            print(f'Previous tunnels: {tunnel}')
        for tunnel in current_tunnels:
            print(f'Current tunnels: {tunnel}')


        nbr_new_tunnels = 0
        for tunnel in current_tunnels:
            nbr_new_tunnels += sum([1 for tun in prev_tunnels if tun.firstHop != tunnel.firstHop or tun.BW != tunnel.BW or tun.fiveQI != tunnel.fiveQI or tun.GW != tunnel.GW or tun.UPF != tunnel.UPF or tun.linkType != tunnel.linkType])
        init_cost = 100*(nbr_new_tunnels)
        print(f'updated {nbr_new_tunnels} tunnels with cost {init_cost}')

    return init_cost