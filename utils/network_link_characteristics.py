import numpy as np
from config import LinkParams

def _mttf(params: LinkParams, distance_m: float) -> float:
    """Distance-dependent MTTF in minutes (Eq. 2)."""
    return params.mttf_ref * (params.d_ref / distance_m) ** params.gamma_fail

def markov_step(state: str, params: LinkParams,
                 distance_m: float, dt_sec: float) -> str:
    """
    One Markov step for a link's UP/DOWN state (Eq. 1).
    All inputs are plain floats.
    """
    
    if state == 'UP':
        mttf   = _mttf(params, distance_m)            
        p_fail = 1.0 - np.exp(-dt_sec / mttf)          
        return 'DOWN' if np.random.random() < p_fail else 'UP'
    
    p_rec  = 1.0 - np.exp(-dt_sec / params.mttr)
    return 'UP' if np.random.random() < p_rec else 'DOWN'

def sample_per(params: LinkParams, distance_m: float,
                utilization: float = 0.0):
    """
    Sample PER ~ Beta(α, β) (Eqs. 3–5).
    All inputs are plain floats.
    """
    mu = (params.per_ref
          * (distance_m / params.d_ref) ** params.gamma_per
          * (1.0 + params.alpha_load * utilization))
    mu = float(np.clip(mu, 0.0, 1.0))

    if mu <= 0.0:
        return 0.0
    if mu >= 1.0:
        return 1.0

    var = (params.cv * mu) ** 2
    var = min(var, 0.999 * mu * (1.0 - mu))   # keep Beta params positive

    conc = mu * (1.0 - mu) / var - 1.0
    return (np.random.beta(mu * conc, (1.0 - mu) * conc))
