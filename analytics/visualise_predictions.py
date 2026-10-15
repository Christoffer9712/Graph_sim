"""
Evaluate trained delay predictors on freshly simulated data.

1. Load the trained topology model (GAT encoder + f_topo).
2. Simulate a new scenario (new start time / seed) with generate_data.
3. Let every model predict the delay of every measured overlay at every step.
4. Print error statistics and plot predictions against the true delay.

Adding another model
--------------------
A model is just a function

    predict(history, step, ovls) -> {ovl_id: predicted delay in ms}

    history : the whole simulated history {step: snapshot}
    step    : the step to predict
    ovls    : the overlays measured at that step (list of dicts with
              'ovl_id', 'ovl_path', 'udl', 'delay_ms')

A model may only use information up to and including what it is allowed to
know at `step` (e.g. persistence only looks at earlier steps). Leave an overlay
out of the returned dict if the model cannot predict it.

Write the function, then add it to the `models` dict in main().
"""
import math

import numpy as np
import torch
import matplotlib.pyplot as plt
from astropy.time import Time
from torch_geometric.data import Data

from algorithm.predictor.path_encoder import HeteroGATv2Encoder
from algorithm.predictor.encoder_util import GraphEncoder
from algorithm.predictor.f_topo import f_topo
from generate_data import generate_data     # <-- adjust to where your generate_data() lives

# Must be the same values as in train.py, otherwise the saved weights don't fit
ENCODER_SETTINGS = dict(node_in=11, virtual_in=1, hidden_dim=128, out_dim=64,
                        heads=2, n_layers=3, dropout=0.1)
FTOPO_SETTINGS = dict(in_dim=64, hidden_dim=64, dropout=0.0)


# =============================================================================
# 1. Loading the trained model
# =============================================================================

def load_topo_model(model_path, device="cpu"):
    """Rebuild encoder + f_topo and load the trained weights."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    encoder = HeteroGATv2Encoder(**ENCODER_SETTINGS).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    encoder.eval()

    ftopo = f_topo(**FTOPO_SETTINGS).to(device)
    ftopo.load_state_dict(checkpoint["ftopo_state"])
    ftopo.eval()

    return {
        "graph_encoder": GraphEncoder(encoder),
        "ftopo": ftopo,
        "mean_delay": float(np.sum(checkpoint["mean_delay"])),   # same normalisation as in training
        "std_delay": float(checkpoint["std_dev_delay"]),
        "device": device,
    }


# =============================================================================
# 2. Simulating test data
# =============================================================================

def simulate_test_data(dataset_path, start_time, seed):
    """Run one new simulation and return its history {step: snapshot}."""
    generate_data(dataset_path, start_times=[start_time], seed=[seed])
    saved = torch.load(dataset_path, weights_only=False)
    return saved["history_list"][0]


def snapshot_to_data(snapshot, device="cpu"):
    """Same conversion as in train.py."""
    return Data(x=snapshot["x"], edge_index=snapshot["edge_index"], edge_attr=snapshot["edge_attr"],
                node_names=snapshot["node_names"], name_to_idx=snapshot["name_to_idx"],
                ovls=snapshot["ovls"]).to(device)


# =============================================================================
# 3. Models  (each: predict(history, step, ovls) -> {ovl_id: delay_ms})
# =============================================================================

def make_topo_predictor(topo_model):
    """Topology-only model: GAT encoder + f_topo on the graph at `step`."""
    def predict(history, step, ovls):
        data = snapshot_to_data(history[step], topo_model["device"])
        with torch.no_grad():
            encodings = topo_model["graph_encoder"].encode_overlays_pyg_batched(data, ovls)
            predictions = {}
            for ovl in ovls:
                out = topo_model["ftopo"](encodings[ovl["ovl_id"]])
                normalised = float(out[0])
                predictions[ovl["ovl_id"]] = normalised * topo_model["std_delay"] + topo_model["mean_delay"]
        return predictions
    return predict


def make_mean_predictor(mean_delay_ms):
    """Baseline: always predict the average delay seen in training."""
    def predict(history, step, ovls):
        return {ovl["ovl_id"]: mean_delay_ms for ovl in ovls}
    return predict


def persistence_predictor(history, step, ovls):
    """Baseline: the delay this overlay had the last time it was measured."""
    earlier_steps = sorted(s for s in history if s < step)
    predictions = {}
    for ovl in ovls:
        for s in reversed(earlier_steps):
            last = [o for o in history[s]["ovls"] if o["ovl_id"] == ovl["ovl_id"]]
            if last:
                predictions[ovl["ovl_id"]] = last[0]["delay_ms"]
                break
    return predictions


# =============================================================================
# 4. Running all models
# =============================================================================

def run_models(history, models):
    """Let every model predict every measured overlay at every step.

    Returns a list of rows: {'step', 'ovl_id', 'true', <model name>: prediction or None}
    """
    rows = []
    for step in sorted(history):
        ovls = history[step]["ovls"]
        if not ovls:
            continue
        predictions = {name: predict(history, step, ovls) for name, predict in models.items()}
        for ovl in ovls:
            row = {"step": step, "ovl_id": ovl["ovl_id"], "true": ovl["delay_ms"]}
            for name in models:
                row[name] = predictions[name].get(ovl["ovl_id"])   # None if the model gave no prediction
            rows.append(row)
    return rows


def print_errors(rows, model_names):
    """RMSE and MAE in ms per model (only over the rows the model predicted)."""
    print(f"\n{'model':<15}{'RMSE [ms]':>12}{'MAE [ms]':>12}{'n':>8}")
    for name in model_names:
        errors = [row[name] - row["true"] for row in rows if row[name] is not None]
        if not errors:
            print(f"{name:<15}{'-':>12}{'-':>12}{0:>8}")
            continue
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        mae = sum(abs(e) for e in errors) / len(errors)
        print(f"{name:<15}{rmse:>12.3f}{mae:>12.3f}{len(errors):>8}")
    print()


# =============================================================================
# 5. Plots
# =============================================================================

def most_measured_overlays(rows, how_many):
    counts = {}
    for row in rows:
        counts[row["ovl_id"]] = counts.get(row["ovl_id"], 0) + 1
    return sorted(counts, key=counts.get, reverse=True)[:how_many]


def plot_over_time(rows, model_names, dt, how_many=4):
    """One panel per overlay: true delay and each model's prediction over time."""
    overlays = most_measured_overlays(rows, how_many)
    fig, axes = plt.subplots(len(overlays), 1, figsize=(10, 2.6 * len(overlays)), sharex=True, squeeze=False)

    for ax, ovl_id in zip(axes[:, 0], overlays):
        ovl_rows = [row for row in rows if row["ovl_id"] == ovl_id]
        times = [row["step"] * dt for row in ovl_rows]

        ax.plot(times, [row["true"] for row in ovl_rows], color="black", marker=".", label="true")
        for name in model_names:
            pts = [(row["step"] * dt, row[name]) for row in ovl_rows if row[name] is not None]
            if pts:
                ax.plot(*zip(*pts), linestyle="--", marker=".", label=name)

        ax.set_title(ovl_id, fontsize=9)
        ax.set_ylabel("delay [ms]")
        ax.grid(alpha=0.3)

    axes[0, 0].legend(fontsize=8)
    axes[-1, 0].set_xlabel("t [s]")
    fig.tight_layout()
    return fig


def plot_predicted_vs_true(rows, model_names):
    """One scatter panel per model; points on the diagonal are perfect predictions."""
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.2 * len(model_names), 4), squeeze=False)
    all_true = [row["true"] for row in rows]
    lo, hi = min(all_true), max(all_true)

    for ax, name in zip(axes[0], model_names):
        pts = [(row["true"], row[name]) for row in rows if row[name] is not None]
        if pts:
            ax.scatter(*zip(*pts), s=8, alpha=0.5)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("true delay [ms]")
        ax.set_ylabel("predicted delay [ms]")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    model_path = "models/predictor_models.pth"
    test_data_path = "data/test_dataset.pt"
    start_time = Time("2026-10-09 09:00:00", scale="utc")   # a time not used for training
    seed = 1234                                              # a seed not used for training
    dt = 8.0                                                 # TIME_STEP used in the simulation

    topo_model = load_topo_model(model_path)
    history = simulate_test_data(test_data_path, start_time, seed)

    # Add new models here: name -> predict(history, step, ovls)
    models = {
        "topo (GAT)": make_topo_predictor(topo_model),
        "train mean": make_mean_predictor(topo_model["mean_delay"]),
        "persistence": persistence_predictor,
#        "oracle prior (path+mean_node_delay)": make_oracle_predictor
    }

    rows = run_models(history, models)
    print_errors(rows, list(models))
    plot_over_time(rows, list(models), dt)
    plot_predicted_vs_true(rows, list(models))
    plt.show()


if __name__ == "__main__":
    main()