"""
Queue-delay diagnostics for DynamicNetwork.

Question we try to answer:
    How persistent is each node's queueing delay, and how much does a
    measurement taken tau seconds ago tell us about the delay now?

Only the queue model is stepped (network.queues.step), not the orbits, so a
long run (many O-U timescales) is cheap. Long runs are needed: with a
correlation time of 1/theta, a run of length T only contains about
T * theta independent samples, and the self-correlation estimate is noisy
unless that number is large.

The network's queue state and random generator are restored afterwards, so
the main simulation is unaffected.

Usage (after building `network`, e.g. before the main loop):
    from queue_delay_test import test_queue_delay_dynamics
    test_queue_delay_dynamics(network, dt=TIME_STEP)
"""
import math

import numpy as np
import matplotlib.pyplot as plt

# Node attributes changed by QueueModel.step
QUEUE_KEYS = ("rho_state", "utilisation", "load", "queue_len", "queue_delay")


# =============================================================================
# Entry point
# =============================================================================

def test_queue_delay_dynamics(network, dt, duration=None, nodes=None,
                              per_type=None, max_lag=None, seed=0,
                              save_dir=None, show=True):
    """Simulate the node queues on their own and plot delay persistence.

    network  : DynamicNetwork (already initialised)
    dt       : step [s], use the simulation's TIME_STEP
    duration : simulated time [s]. Default 100 / theta (100 correlation times)
    nodes    : list of node ids to study. Default: picked with `per_type`
    per_type : {node_type: count}, used when `nodes` is None
    max_lag  : largest lag [s] in the self-correlation plots. Default 3 / theta
    save_dir : if given, figures are also saved there as PNG

    Returns a dict with times, delays, utilisations and self-correlations.
    """
    queue = network.queues
    theta = queue.ou_theta
    if duration is None:
        duration = 100.0 / theta
    n_steps = int(math.ceil(duration / dt))
    if max_lag is None:
        max_lag = 3.0 / theta
    max_lag_steps = max(1, min(int(round(max_lag / dt)), n_steps // 4))

    if nodes is None:
        nodes = pick_nodes(network, per_type, seed)
    if not nodes:
        print("No nodes to study.")
        return None

    # 1. Run the queues and record delay [ms] and utilisation for each node
    saved = save_queue_state(network)
    try:
        delays = np.empty((len(nodes), n_steps))
        rhos = np.empty((len(nodes), n_steps))
        for k in range(n_steps):
            queue.step(dt)
            for i, n in enumerate(nodes):
                d = network.graph.nodes[n]
                delays[i, k] = d["queue_delay"] * 1000.0
                rhos[i, k] = d["utilisation"]
    finally:
        restore_queue_state(network, saved)

    times = np.arange(1, n_steps + 1) * dt
    lags_s = np.arange(max_lag_steps + 1) * dt

    # 2. Self-correlation of each node's delay
    acfs = np.array([self_correlation(x, max_lag_steps) for x in delays])

    # 3. Summary table + plots
    print_summary(network, nodes, delays, rhos, acfs, dt, theta)

    # Show only the first ~10 correlation times over time; the full run is
    # too dense to read (all of it is still used for the statistics).
    shown = times <= min(duration, 10.0 / theta)
    figs = [
        plot_delay_and_utilisation(network, nodes, times[shown],
                                   delays[:, shown], rhos[:, shown]),
        plot_self_correlation(nodes, lags_s, acfs, theta, dt),
        plot_value_of_past_measurement(nodes, lags_s, acfs, theta),
    ]
    if save_dir is not None:
        for fig, name in zip(figs, ["queue_delay_over_time",
                                    "queue_delay_self_correlation",
                                    "queue_delay_past_measurement"]):
            fig.savefig(f"{save_dir}/{name}.png", dpi=130)
    if show:
        plt.show()

    return {"nodes": nodes, "times": times, "delays_ms": delays,
            "utilisation": rhos, "lags_s": lags_s, "self_correlation": acfs}


# =============================================================================
# Helpers
# =============================================================================

def pick_nodes(network, per_type=None, seed=0):
    """Pick a few nodes of each type. Satellites are taken from the European
    region so they are ones that actually appear in overlay paths."""
    if per_type is None:
        per_type = {"aircraft": 1, "satellite": 3, "gateway": 1, "upf": 1}
    rng = np.random.default_rng(seed)

    chosen = []
    for node_type, count in per_type.items():
        candidates = sorted(
            n for n, d in network.graph.nodes(data=True)
            if d.get("node_type") == node_type
            and (node_type != "satellite" or n in network.topology.euro_nodes)
        )
        if not candidates:
            print(f"No nodes of type '{node_type}' found, skipping.")
            continue
        picks = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        chosen += [candidates[i] for i in sorted(picks)]
    return chosen


def save_queue_state(network):
    node_state = {n: {k: d[k] for k in QUEUE_KEYS if k in d}
                  for n, d in network.graph.nodes(data=True)}
    return node_state, network.queues.rng.bit_generator.state


def restore_queue_state(network, saved):
    node_state, rng_state = saved
    for n, values in node_state.items():
        network.graph.nodes[n].update(values)
    network.queues.rng.bit_generator.state = rng_state


def self_correlation(x, max_lag_steps):
    """Standard autocorrelation estimate: one mean and one variance for the
    whole series, so values at different lags are directly comparable."""
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0:
        return np.full(max_lag_steps + 1, np.nan)
    return np.array([np.dot(x[:n - k], x[k:]) / (n * var)
                     for k in range(max_lag_steps + 1)])


def first_lag_below(acf, level, dt):
    below = np.nonzero(acf < level)[0]
    return below[0] * dt if below.size else np.nan


def print_summary(network, nodes, delays, rhos, acfs, dt, theta):
    print(f"\nO-U: theta = {theta:g} 1/s (timescale 1/theta = {1/theta:.0f} s), "
          f"sigma = {network.queues.ou_sigma:g}, "
          f"stationary std of utilisation = {network.queues.ou_sigma / math.sqrt(2 * theta):.3f}")
    print("Linear model predicts self-correlation exp(-theta * lag): "
          f"drops below 0.5 at {math.log(2) / theta:.0f} s, below 1/e at {1/theta:.0f} s\n")

    header = (f"{'node':<22}{'type':<10}{'mu':>7}{'rho_bar':>8}"
              f"{'W(rho_bar)':>11}{'mean':>8}{'std':>7}"
              f"{'@max':>7}{'@min':>7}{'<0.5':>8}{'<1/e':>8}")
    print(header)
    print(f"{'':<22}{'':<10}{'[1/s]':>7}{'':>8}{'[ms]':>11}{'[ms]':>8}{'[ms]':>7}"
          f"{'[%]':>7}{'[%]':>7}{'[s]':>8}{'[s]':>8}")
    print("-" * len(header))
    for i, n in enumerate(nodes):
        d = network.graph.nodes[n]
        mu, rho_bar = d["service_rate"], d["base_rho"]
        w_lin = 1000.0 / (mu * (1.0 - rho_bar))   # delay at the O-U mean
        at_max = 100 * np.mean(np.isclose(rhos[i], network.queues.rho_max))
        at_min = 100 * np.mean(np.isclose(rhos[i], network.queues.rho_min))
        print(f"{n:<22}{d['node_type']:<10}{mu:>7.0f}{rho_bar:>8.2f}"
              f"{w_lin:>11.2f}{delays[i].mean():>8.2f}{delays[i].std():>7.2f}"
              f"{at_max:>7.1f}{at_min:>7.1f}"
              f"{first_lag_below(acfs[i], 0.5, dt):>8.0f}"
              f"{first_lag_below(acfs[i], 1 / math.e, dt):>8.0f}")
    print("\n@max / @min: share of time the utilisation sits at the clamp. "
          "W(rho_bar) vs mean: bias of the linearised mean delay.\n")


# =============================================================================
# Plots
# =============================================================================

def plot_delay_and_utilisation(network, nodes, times, delays, rhos):
    """Queueing delay and utilisation over time, one line per node."""
    fig, (ax_d, ax_r) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colours = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, n in enumerate(nodes):
        label = f"{n} ({network.graph.nodes[n]['node_type']})"
        ax_d.plot(times, delays[i], color=colours[i % 10], linewidth=0.8, label=label)
        ax_r.plot(times, rhos[i], color=colours[i % 10], linewidth=0.8)

    ax_d.set_ylabel("queueing delay [ms]")
    ax_d.set_yscale("log")
    ax_d.set_title("Node queueing delay over time")
    ax_d.legend(fontsize=8, ncol=2)
    ax_d.grid(alpha=0.3)

    for level in (network.queues.rho_min, network.queues.rho_max):
        ax_r.axhline(level, color="grey", linestyle="--", linewidth=0.8)
    ax_r.set_ylabel("utilisation ρ")
    ax_r.set_xlabel("t [s]")
    ax_r.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_self_correlation(nodes, lags_s, acfs, theta, dt):
    """Self-correlation of each node's delay against the linear-model curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, n in enumerate(nodes):
        ax.plot(lags_s, acfs[i], linewidth=1.2, label=n)
    ax.plot(lags_s, np.exp(-theta * lags_s), color="black", linestyle="--",
            linewidth=2, label="exp(-θ·lag) (linear model)")
    ax.axvline(dt, color="grey", linewidth=0.8, linestyle=":")
    ax.text(dt, 0.02, " one step", color="grey", fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("lag [s]")
    ax.set_ylabel("self-correlation of delay")
    ax.set_title("How long a node's delay stays similar to itself")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_value_of_past_measurement(nodes, lags_s, acfs, theta):
    """Prediction error when the only information is one measurement of the
    same node taken `lag` seconds ago, relative to knowing nothing (MSE / var).

    best linear  : mean + ρ(lag)·(x_old − mean)  →  1 − ρ(lag)²
    persistence  : x_old                         →  2·(1 − ρ(lag))
    prior mean   : mean                          →  1
    Persistence is worse than the prior mean once ρ(lag) < 0.5.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, n in enumerate(nodes):
        c = colours[i % 10]
        ax.plot(lags_s, 1 - acfs[i] ** 2, color=c, linewidth=1.2, label=n)
        ax.plot(lags_s, 2 * (1 - acfs[i]), color=c, linewidth=0.8, linestyle="--")

    rho_lin = np.exp(-theta * lags_s)
    ax.plot(lags_s, 1 - rho_lin ** 2, color="black", linewidth=2,
            label="linear model: best linear")
    ax.plot(lags_s, 2 * (1 - rho_lin), color="black", linewidth=2, linestyle="--",
            label="linear model: persistence")
    ax.axhline(1, color="grey", linewidth=1.5, label="prior mean only")

    ax.set_ylim(0, 1.6)
    ax.set_xlabel("age of the measurement [s]")
    ax.set_ylabel("MSE / variance")
    ax.set_title("Value of one past measurement of the same node\n"
                 "(solid: best linear predictor, dashed: persistence)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig