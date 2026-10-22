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
from algorithm.predictor.f_cov import f_cov
from generate_data import generate_data     # <-- adjust to where your generate_data() lives

# Must be the same values as in train.py, otherwise the saved weights don't fit
ENCODER_SETTINGS = dict(node_in=10, virtual_in=1, hidden_dim=128, out_dim=64,
                        heads=2, n_layers=3, dropout=0.1)
FTOPO_SETTINGS = dict(in_dim=64, hidden_dim=64, dropout=0.0)
FCOV_SETTINGS = dict(in_dim=64, dropout=0.0)


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

def load_topo_cov_model(model_path, device="cpu"):
    """Rebuild encoder + f_topo and load the trained weights."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    encoder = HeteroGATv2Encoder(**ENCODER_SETTINGS).to(device)
    encoder.load_state_dict(checkpoint["encoder_state"])
    encoder.eval()

    ftopo = f_topo(**FTOPO_SETTINGS).to(device)
    ftopo.load_state_dict(checkpoint["ftopo_state"])
    ftopo.eval()

    fcov = f_cov(**FCOV_SETTINGS).to(device)
    fcov.load_state_dict(checkpoint["fcov_state"])
    fcov.eval()
    
    return {
        "graph_encoder": GraphEncoder(encoder),
        "ftopo": ftopo,
        "fcov": fcov,
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

def make_topo_cov_predictor(model, Delta=3, M=1, use_own_measurements=True, use_only_own_measurements=False):
    """Topology + covariance model, the same computation as train.py with INCLUDE_MEAS=True:

        prediction = f_topo(enc) + g^T C^-1 (meas - f_topo)      (in normalised units)

    using the measurements of all overlays in the `Delta` steps that end M steps
    before `step` (training: history t, t-1, ..., t-Delta+1 -> target t+M).
    Delta and M must be the values used in training.
    use_own_measurements=False drops the overlay's own past measurements
    (like REMOVE_SELF_MEAS in train.py).
    """
    device = model["device"]
    graph_encoder, ftopo, fcov = model["graph_encoder"], model["ftopo"], model["fcov"]
    mean, std = model["mean_delay"], model["std_delay"]

    # Integer keys for fcov, built exactly as in train.py (only equality between keys matters)
    name_vocab, id_vocab = {}, {}
    def ovl_keys(ovl):
        wps = [name_vocab.setdefault(nm, len(name_vocab)) for nm in ovl['ovl_path']][:fcov.n_wp]
        wps += [-1] * (fcov.n_wp - len(wps))
        return torch.tensor([id_vocab.setdefault(ovl['ovl_id'], len(id_vocab))] + wps,
                            dtype=torch.long, device=device)

    # Encodings never change (model in eval mode), so each (step, overlay) is encoded once
    cache = {"history": None, "steps": {}}
    def encode_step(history, s, ovls):
        """{ovl_id: (encoding, normalised f_topo output)} for the given overlays at step s."""
        if cache["history"] is not history:              # new simulation -> empty the cache
            cache["history"], cache["steps"] = history, {}
        step_cache = cache["steps"].setdefault(s, {})
        missing = [o for o in ovls if o["ovl_id"] not in step_cache]
        if missing:
            data = snapshot_to_data(history[s], device)
            encodings = graph_encoder.encode_overlays_pyg_batched(data, missing)
            for o in missing:
                enc = encodings[o["ovl_id"]]
                step_cache[o["ovl_id"]] = (enc, ftopo(enc))
        return step_cache

    def predict(history, step, ovls):
        with torch.no_grad():
            # 1. Past measurements of all overlays (the "history" in train.py)
            H_hist, z_hist, elapsed, keys_hist, ids_hist = [], [], [], [], []
            for i in range(Delta):
                s = step - M - i
                if s not in history:
                    continue
                past_ovls = history[s]["ovls"]
                encoded = encode_step(history, s, past_ovls)
                for o in past_ovls:
                    enc, out_topo = encoded[o["ovl_id"]]
                    meas = (o["delay_ms"] - mean) / (std + 1e-8)
                    H_hist.append(enc)
                    z_hist.append(meas - out_topo[0])          # measured minus topology part
                    elapsed.append(float(i))
                    keys_hist.append(ovl_keys(o))
                    ids_hist.append(o["ovl_id"])

            if H_hist:
                H_hist = torch.stack(H_hist)
                z_hist = torch.stack(z_hist)
                lag = torch.tensor(elapsed, device=device).unsqueeze(1) + M
                keys_hist = torch.stack(keys_hist)

            # 2. Prediction for every overlay at `step`
            current = encode_step(history, step, ovls)
            predictions = {}
            for ovl in ovls:
                enc, out_topo = current[ovl["ovl_id"]]
                normalised = out_topo[0]

                if len(H_hist) > 0:
                    if use_only_own_measurements:
                        use = torch.tensor([hid == ovl["ovl_id"] for hid in ids_hist],
                                       dtype=torch.bool, device=device)
                    else:
                        use = torch.tensor([use_own_measurements or hid != ovl["ovl_id"] for hid in ids_hist],
                                       dtype=torch.bool, device=device)
                    n = int(use.sum())
                    if n > 0:
                        #C = fcov(H_hist[use], lag[use], keys_hist=keys_hist[use],
                        #         include_phi=False, add_noise=False).reshape(n, n)
                        #g = fcov(H_hist[use], lag[use], h_pred=enc, keys_hist=keys_hist[use],
                        #         keys_pred=ovl_keys(ovl), include_phi=False, add_noise=False).reshape(n, 1)
                        C = fcov(H_hist[use], lag[use], include_phi=True, add_noise=False).reshape(n, n)
                        g = fcov(H_hist[use], lag[use], h_pred=enc, include_phi=True, add_noise=False).reshape(n, 1)
                        R = 0.001*torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
                        correction = g.T @ torch.linalg.solve(C+R, z_hist[use])
                        normalised = normalised + correction[0]

                predictions[ovl["ovl_id"]] = float(normalised) * std + mean
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

def analytic_prior_predictor(history, step, ovls):
    return {ovl['ovl_id']: ovl['prior_delay_ms'] for ovl in ovls}

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
    #model_path = "models/predictor_models.pth"
    model_path = "models/predictor_models_trained_with_meas_overlays_in_succession.pth"
    test_data_path = "data/test_dataset.pt"
    start_time = Time("2026-10-09 09:00:00", scale="utc")   # a time not used for training
    seed = 1234                                              # a seed not used for training
    dt = 8.0                                                 # TIME_STEP used in the simulation

    topo_model = load_topo_cov_model(model_path)
    history = simulate_test_data(test_data_path, start_time, seed)

    models = {
        "topo (GAT)": make_topo_predictor(topo_model),
        "topo + cov": make_topo_cov_predictor(topo_model),
        "topo + cov (no own)": make_topo_cov_predictor(topo_model, use_own_measurements=False),
        "topo + cov (only own)": make_topo_cov_predictor(topo_model, use_only_own_measurements=True),
        "train mean": make_mean_predictor(topo_model["mean_delay"]),
        "persistence": persistence_predictor,
        "oracle prior": analytic_prior_predictor,
    }

    rows = run_models(history, models)
    print_errors(rows, list(models))
    plot_over_time(rows, list(models), dt)
    plot_predicted_vs_true(rows, list(models))
    plt.show()

if __name__ == "__main__":
    main()