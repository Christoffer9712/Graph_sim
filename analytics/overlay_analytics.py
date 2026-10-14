"""
Overlay delay analytics.

Question we try to answer:
    How much does knowing the delay of overlay j at time t tell us about
    the delay of overlay i at a later time t + lag?

Usage (from the main script):
    analyse_overlay_delays(overlay_records, dt)

overlay_records is a list of dicts, one per measurement:
    {'t': time [s], 'overlay': overlay path, 'udl': underlay path, 'delay_ms': delay [ms]}
"""
import numpy as np
import matplotlib.pyplot as plt

# If two series share fewer measured time steps than this, we don't trust the
# statistic and report NaN instead.
MIN_PAIRS = 10


# =============================================================================
# Entry point
# =============================================================================

def analyse_overlay_delays(records, dt, min_samples=20, max_lag_steps=30):
    """Turn the raw measurements into time series and make all plots."""

    # 1. One delay time series per overlay (NaN where not measured)
    times, delays = build_delay_series(records, dt)
    delays = keep_frequent_overlays(delays, min_samples)
    if not delays:
        print("Not enough overlay measurements for statistics "
              "(lower min_samples or extend SIM_DURATION).")
        return

    # 2. Replace long overlay paths with short labels O0, O1, ... (legend is printed)
    delays = rename_to_short_labels(delays)

    # 3. Change in delay from one step to the next. This removes the slow trend
    #    that all overlays share (the aircraft moving), which would otherwise
    #    make every overlay look correlated with every other one.
    delay_changes = {name: change_per_step(d) for name, d in delays.items()}

    # 4. Pick which overlay to study in detail, and how far ahead to look.
    #    Beyond ~1/4 of the series there are too few overlapping samples.
    target = overlay_with_most_samples(delays)
    max_lag_steps = max(1, min(max_lag_steps, len(times) // 4))

    # 5. Plots
    plot_delays_over_time(times, delays)

    plot_heatmap(delays, lag=0, kind="cor",
                 title="Correlation of delay, same time step")

    plot_heatmap(delays, lag=1, kind="cor",
                     title="Correlation of delay [ms²], lag 1")
    
    
    plot_self_correlation(delays[target], target, max_lag_steps, dt)

    plot_prediction_curves(delay_changes, target, max_lag_steps, dt)

    plt.show()


# =============================================================================
# Data preparation
# =============================================================================

def build_delay_series(records, dt):
    """Turn the list of measurements into one delay time series per overlay.

    Returns
        times  : array of simulation times [s], one per time step
        delays : {overlay name: array of delays [ms]}, NaN at time steps
                 where the overlay was not measured
    """
    if not records:
        return np.array([]), {}

    n_steps = round(max(r["t"] for r in records) / dt) + 1
    times = np.arange(n_steps) * dt

    delays = {}
    for r in records:
        name = " → ".join(str(node) for node in r["overlay"])
        step = round(r["t"] / dt)

        if name not in delays:
            delays[name] = np.full(n_steps, np.nan)   # start with "not measured"
        delays[name][step] = r["delay_ms"]

    return times, delays


def count_samples(x):
    """Number of time steps where x was actually measured (not NaN)."""
    return int(np.sum(~np.isnan(x)))


def keep_frequent_overlays(delays, min_samples):
    """Drop overlays measured too rarely to give meaningful statistics."""
    return {name: d for name, d in delays.items() if count_samples(d) >= min_samples}


def rename_to_short_labels(delays):
    """Rename overlays to O0, O1, ... and print which label is which path."""
    renamed = {}
    for i, (name, d) in enumerate(delays.items()):
        label = f"O{i}"
        print(f"{label}: {name}  ({count_samples(d)} samples)")
        renamed[label] = d
    return renamed


def change_per_step(x):
    """Delay change between consecutive steps: x[t] - x[t-1]. First value is NaN."""
    change = np.full(len(x), np.nan)
    change[1:] = x[1:] - x[:-1]
    return change


def overlay_with_most_samples(delays):
    return max(delays, key=lambda name: count_samples(delays[name]))


# =============================================================================
# Statistics
# =============================================================================

def lagged_statistic(x, y, lag, kind="corr"):
    """Covariance or correlation between x at time t and y at time t + lag.

    lag is in time steps. lag = 0 compares the two series at the same time.
    Only time steps where both values were measured are used.
    """
    x_now = x[:len(x) - lag]     # x(t)        for t = 0 ... end - lag
    y_later = y[lag:]            # y(t + lag)  for the same t

    both_measured = ~np.isnan(x_now) & ~np.isnan(y_later)
    if np.sum(both_measured) < MIN_PAIRS:
        return np.nan

    a = x_now[both_measured]
    b = y_later[both_measured]

    if kind == "cov":
        return np.cov(a, b)[0, 1]

    # Correlation is undefined if one series doesn't vary at all
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return np.corrcoef(a, b)[0, 1]


# =============================================================================
# Plots
# =============================================================================

def plot_delays_over_time(times, delays):
    """Raw delay of every overlay. Gaps = overlay not measured at that step."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, d in delays.items():
        ax.plot(times, d, marker=".", label=name)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("delay [ms]")
    ax.set_title("Overlay delay over time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_heatmap(series, lag, kind, title):
    """Matrix of lagged_statistic for every pair of overlays.
    Row i = overlay i at time t, column j = overlay j at time t + lag."""
    names = list(series)
    n = len(names)

    matrix = np.full((n, n), np.nan)
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            matrix[i, j] = lagged_statistic(series[name_i], series[name_j], lag, kind)

    # Symmetric colour scale around zero (red = positive, blue = negative)
    if kind == "corr" or np.all(np.isnan(matrix)):
        limit = 1.0
    else:
        limit = np.nanmax(np.abs(matrix))

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit)

    # Write the value in each cell
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    ax.set_xticks(range(n))
    ax.set_xticklabels(names)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    ax.set_ylabel("overlay i at t")
    ax.set_xlabel("overlay j at t + lag")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()


def plot_self_correlation(x, name, max_lag_steps, dt):
    """How similar an overlay's delay is to its own delay `lag` seconds later.
    (Self-covariance has the same shape; it is this curve times the variance.)"""
    lags = list(range(max_lag_steps + 1))
    lag_seconds = [lag * dt for lag in lags]
    values = [lagged_statistic(x, x, lag, "corr") for lag in lags]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lag_seconds, values, marker="o")
    ax.axhline(0, color="grey", linewidth=0.8)

    # Values inside the grey band can't be told apart from zero (rough 95% bound)
    noise_level = 1.96 / np.sqrt(count_samples(x))
    ax.axhspan(-noise_level, noise_level, color="grey", alpha=0.2)

    ax.set_xlabel("lag [s]")
    ax.set_ylabel("self-correlation")
    ax.set_title(f"Self-correlation of {name} delay")
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_prediction_curves(series, target, max_lag_steps, dt):
    """For every overlay j: correlation between j now and the target in the future.

    The thick black line is the target's own past. Another overlay only tells
    you something extra about the target's future where its line is above it.
    """
    lags = list(range(max_lag_steps + 1))
    horizon_seconds = [lag * dt for lag in lags]

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, x in series.items():
        values = [lagged_statistic(x, series[target], lag, "corr") for lag in lags]
        if name == target:
            ax.plot(horizon_seconds, values, color="black", linewidth=3,
                    label=f"{name} (own past)")
        else:
            ax.plot(horizon_seconds, values, marker=".", label=name)

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("prediction horizon [s]")
    ax.set_ylabel(f"correlation with future {target}")
    ax.set_title(f"How well each overlay's delay change predicts {target}'s future delay change")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()