"""
Per-node queueing delay: Ornstein-Uhlenbeck utilisation + M/M/1 delay.

Only reads/writes the queue attributes of the graph nodes
(service_rate, base_rho, rho_state, utilisation, load, queue_len, queue_delay).
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class QueueParams:
    service_rate_range: tuple = (0.8, 1.2)    # mu, per node type below
    base_load_range: tuple = (0.3, 0.6)       # mean rho each node sits at
    ou_theta: float = 0.006                   # mean-reversion rate (1/timescale)
    ou_sigma: float = 0.01                    # driving noise magnitude
    rho_max: float = 0.9                      # clamp to keep M/M/1 finite
    rho_min: float = 0.02                     # lower clamp of the utilisation
    burn_in_time: float | None = None         # extra warm-up before t=0 (None -> 2 / ou_theta)
    burn_in_dt: float = 1.0                   # step used during the warm-up (use the simulation dt)


class QueueModel:
    def __init__(self, graph, params: QueueParams | None = None, seed: int = 42):
        self.graph = graph
        self.params = params if params is not None else QueueParams()
        self.rng = np.random.default_rng(seed)

        self.ou_theta = self.params.ou_theta
        self.ou_sigma = self.params.ou_sigma
        self.rho_max = self.params.rho_max
        self.rho_min = self.params.rho_min
        self.type_mu = {'aircraft':  400,    # access link: ~5 ms (the real bottleneck)
                        'satellite': 1000,
                        'gateway':   1000,
                        'aviation':  1000,
                        'upf':       1000}

    # ----Public----
    def init(self):
        """Draw per-node parameters and initial state, then burn in. Call once all nodes exist."""
        p = self.params
        for _, d in self.graph.nodes(data=True):
            mult = self.type_mu.get(d["node_type"])
            mu = self.rng.uniform(*p.service_rate_range) * mult
            rho0 = self.rng.uniform(*p.base_load_range)
            d["service_rate"] = mu
            d["base_rho"] = rho0          # OU mean, in utilisation units

            # Stationary std of the (unclipped) O-U utilisation: sigma / sqrt(2 theta)
            stat_std = self.ou_sigma / math.sqrt(2.0 * self.ou_theta)

            # Initial state: a draw from the stationary distribution N(rho0, sigma^2 / (2 theta))
            # instead of the mean itself. Starting at the mean gives zero variance at t=0
            # that only builds up over ~1/(2 theta) steps, so E[queue_delay] (convex in rho)
            # would drift upwards during the simulation.
            rho_init = float(np.clip(rho0 + stat_std * self.rng.standard_normal(), self.rho_min, self.rho_max))
            d["rho_state"] = rho_init     # OU state
            d["utilisation"] = rho_init
            d["load"] = rho_init * mu     # lambda
            d["queue_len"] = self._equilibrium_len(rho_init, rho_max=self.rho_max)
            d["queue_delay"] = self._equilibrium_delay(rho_init, mu, rho_max=self.rho_max)

        # Warm-up: the clamp to [rho_min, rho_max] makes the true stationary distribution
        # slightly non-Gaussian; running the process for a while before t=0 removes the
        # remaining mismatch (it decays like exp(-theta * t)).
        burn_in_time = p.burn_in_time if p.burn_in_time is not None else 2.0 / self.ou_theta
        for _ in range(int(math.ceil(burn_in_time / p.burn_in_dt))):
            self.step(p.burn_in_dt)

    def step(self, dt):
        """Advance all node queues by dt."""
        for n, d in self.graph.nodes(data=True):
            mu = d["service_rate"]

            # --- 1. Offered load: Ornstein-Uhlenbeck, mean-reverting ---------
            mean = d["base_rho"]
            drift = self.ou_theta * (mean - d["rho_state"]) * dt
            noise = self.ou_sigma * math.sqrt(dt) * self.rng.standard_normal()
            rho = d["rho_state"] + drift + noise
            rho = float(np.clip(rho, self.rho_min, self.rho_max))
            d["rho_state"] = rho
            d["utilisation"] = rho
            d["load"] = rho * mu

            # --- 2. Queue length at equilibrium ------------------------------
            d["queue_len"] = self._equilibrium_len(rho, self.rho_max)

            # --- 3. Derived metrics ------------------------------------------
            # Delay from actual occupancy via Little's law: W = L / lambda
            d["queue_delay"] = d["queue_len"] / max(d["load"], 1e-6)

    # ---Helpers---
    @staticmethod
    def _equilibrium_len(rho, rho_max):
        """L = rho / (1 - rho)."""
        rho = min(rho, rho_max)
        return rho / (1.0 - rho)

    @staticmethod
    def _equilibrium_delay(rho, mu, rho_max):
        """W = 1 / (mu - lambda) = 1 / (mu (1 - rho))."""
        rho = min(rho, rho_max)
        return 1.0 / (mu * (1.0 - rho))