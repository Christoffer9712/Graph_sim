"""
Train MeasurementEmbedder + Predictor on generated dataset.

Saves the trained weights to models/predictor_embedder.pth and
models/predictor_predictor.pth (two modules).
"""
import os
import torch
import copy
import numpy as np
import random

from algorithm.predictor.path_encoder import HeteroGATv2Encoder
from algorithm.predictor.encoder_util import GraphEncoder
from torch_geometric.data import Data
from .dataset import PredictorDataset
from algorithm.predictor.f_topo import f_topo
from algorithm.predictor.f_cov import f_cov

GRAD_CLIP = 1.0
INCLUDE_MEAS = True
REMOVE_SELF_MEAS = True
REMOVE_SELF_MEAS_PERC = 1.0

def train(dataset_path=None, model_dir=None, epochs=50, batch_size=32, lr=0.5e-3,
          device='cpu', min_delta=1e-4, patience=10, transfer_learning_model=None):

    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictor_dataset.pt')
        dataset_path = os.path.abspath(dataset_path)
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    dataset = PredictorDataset(dataset_path)
    # Loss weights: c_delta
    c_delta = 1.0

    nbr_sims = dataset.get_nbr_of_sims()

    # Prediction horizon: M
    M = 1 # NOTE!!!

    # Measurement history depth: Delta
    Delta = 3

    n_steps = len(dataset[0])
    # Contiguous time-based split (no shuffling across splits -> no leakage)
    lo, hi = Delta - 1, n_steps - M
    all_t = list(range(lo, hi))
    n = len(all_t)
    gap = M
    train_t = all_t[int(0.10*n)+gap:int(0.30*n)] + all_t[int(0.40*n)+gap:int(0.70*n)] + all_t[int(0.80*n)+gap:]
    val_t   = all_t[:int(0.10*n)-gap] + all_t[int(0.30*n)+gap:int(0.40*n)-gap] + all_t[int(0.70*n)+gap:int(0.80*n)-gap]

    test_t  = all_t # Used on unseen topology
    (mean_delay, std_dev_delay) = calculate_mean_std(train_t, dataset[:-1], nbr_sims-1)

    #print(f"Baseline (predict-mean) train = {baselines(train_t, dataset[:-1], nbr_sims-1, mean_delay.sum(), std_dev_delay, c_delta)}")
    #print(f"Baseline (predict-mean) val   = {baselines(val_t, dataset[:-1], nbr_sims-1, mean_delay.sum(), std_dev_delay, c_delta)}")
    #print(f"Baseline (predict-mean) test  = {baselines(test_t, [dataset[-1]], 1, mean_delay.sum(), std_dev_delay, c_delta)}")

    TOPO_DIM = 64
    encoder = HeteroGATv2Encoder(node_in=10, virtual_in=1, hidden_dim=128, out_dim=TOPO_DIM, heads=2, n_layers=3, dropout=0.0).to(device)
    graph_encoder = GraphEncoder(encoder)

    fcov = f_cov(in_dim=TOPO_DIM, dropout=0.0).to(device)
    ftopo = f_topo(in_dim=TOPO_DIM, hidden_dim=64, dropout=0.0).to(device)

    opt = torch.optim.Adam(list(encoder.parameters()) + list(ftopo.parameters()) + list(fcov.parameters()), lr=lr)
    
    if transfer_learning_model:
        model_path = os.path.join(model_dir, transfer_learning_model)
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        ftopo.load_state_dict(ckpt['ftopo_state'])
        encoder.load_state_dict(ckpt['encoder_state'])
        fcov.load_state_dict(ckpt['fcov_state'])
        print(f"Loaded model at {model_path}")
        opt = torch.optim.Adam([
            {'params': encoder.parameters(), 'lr': 0.2*lr},
            {'params': ftopo.parameters(), 'lr': 0.2*lr},
            {'params': fcov.parameters(), 'lr': lr}
        ])
        mean_delay, std_dev_delay = ckpt['mean_delay'], ckpt['std_dev_delay']
        print(f'Loaded model from {transfer_learning_model}')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.5, patience=3)

    name_vocab, id_vocab = {}, {}
    def ovl_keys(ovl):
        wps = [name_vocab.setdefault(nm, len(name_vocab)) for nm in ovl['ovl_path']][:fcov.n_wp]
        wps += [-1] * (fcov.n_wp - len(wps))
        return torch.tensor([id_vocab.setdefault(ovl['ovl_id'], len(id_vocab))] + wps, dtype=torch.long, device=device)

    def run_t(t, gidx, data, h_enc, h_topo):
        curr_overlay = data[(gidx,t)]['ovls']
        curr_overlay_ids = [ovl['ovl_id'] for ovl in curr_overlay]

        loss = torch.zeros((), device=device)
        count = 0

        H_hist = []
        Meas_hist = []
        Elapsed = []
        Hist_ids = []
        Topo_hist = []
        Keys_hist = []
        if INCLUDE_MEAS:
            for i in range(0, Delta):
                d = dataset[gidx][t - i]
                hist_overlay = d['ovls']
                # encode the whole graph for time (t-i) ONCE, for all overlays
                missing = [ovl for ovl in hist_overlay if (gidx, ovl['ovl_id'], t - i) not in h_enc]
                if missing:
                    enc = graph_encoder.encode_overlays_pyg_batched(data[(gidx, t - i)], missing)
                    topo = ftopo(torch.stack([enc[i] for i in list(enc.keys())]))
                    for idx, entry in enumerate(enc.keys()):
                        h_enc[(gidx, entry, t - i)] = enc[entry]
                        h_topo[(gidx, entry, t - i)] = topo[idx]
                        
                for ovl in hist_overlay:
                    H_hist.append(h_enc[(gidx, ovl['ovl_id'], t - i)])
                    Meas_hist.append(torch.tensor(ovl['delay_ms'], dtype=torch.float, device=device))
                    Topo_hist.append(h_topo[(gidx, ovl['ovl_id'], t-i)])
                        
                    Elapsed.append(float(i))
                    Hist_ids.append((ovl['ovl_id'], t - i))
                    Keys_hist.append(ovl_keys(ovl))

            if len(H_hist) > 0:  # Can be 0 if there are no overlays at given time
                H_hist = torch.stack(H_hist)
                Meas_hist = torch.stack(Meas_hist)
                Meas_hist = torch.stack([(Meas_hist - mean_delay) / (std_dev_delay + 1e-8)], dim=1)
                Elapsed = torch.tensor(Elapsed, device=device).unsqueeze(1)
                Topo_hist = torch.stack(Topo_hist)
                Keys_hist = torch.stack(Keys_hist)

        for m in range(1, M+1):         
            fut_overlay = data[(gidx, t + m)]['ovls']

            # overlays we actually train on this step
            eligible = [f_ovl for f_ovl in fut_overlay
                        if f_ovl['ovl_id'] in curr_overlay_ids]
            if not eligible:
                continue

            enc = graph_encoder.encode_overlays_pyg_batched(data[(gidx, t + m)], eligible)
            for entry in enc.keys():
                h_enc[(gidx, entry, t + m)] = enc[entry]

            for f_ovl in eligible:
                tgt = torch.tensor(f_ovl['delay_ms'], dtype=torch.float, device=device)
                tgt = (tgt - mean_delay) / (std_dev_delay + 1e-8)

                ovl_id = f_ovl['ovl_id']
                enc = h_enc[(gidx, ovl_id, t + m)]
                out_topo = ftopo(enc)
                h_topo[(gidx, ovl_id, t+m)] = out_topo

                if INCLUDE_MEAS:
                    mask = torch.ones((len(H_hist)), dtype=torch.bool, device=device).detach()
                    if REMOVE_SELF_MEAS:
                        for d in range(Delta):  #Don't use previous measurements from the same overlay
                            if (f_ovl['ovl_id'], t-d) in Hist_ids:
                                if random.random() < REMOVE_SELF_MEAS_PERC:
                                    idx = Hist_ids.index((f_ovl['ovl_id'], t-d))
                                    mask[idx] = False

                    Topo_hist_tmp = Topo_hist[mask]
                    Elapsed_tmp = Elapsed[mask]
                    H_hist_tmp = H_hist[mask]
                    Meas_hist_tmp = Meas_hist[mask]
                    Keys_hist_tmp = Keys_hist[mask]

                    # Only use shared overlay information
                    #C = (fcov(H_hist_tmp, Elapsed_tmp + m, keys_hist=Keys_hist_tmp, include_phi=False, add_noise=False)).reshape(H_hist_tmp.size(0), H_hist_tmp.size(0))
                    #g = (fcov(H_hist_tmp, Elapsed_tmp + m, h_pred=enc,
                    #          keys_hist=Keys_hist_tmp, keys_pred=ovl_keys(f_ovl), include_phi=False, add_noise=False)).reshape(H_hist_tmp.size(0), 1)
                    
                    # Only use the phi function to estimate covariance based on encodings
                    C = (fcov(H_hist_tmp, Elapsed_tmp + m, include_phi=True, add_noise=False)).reshape(H_hist_tmp.size(0), H_hist_tmp.size(0))
                    g = (fcov(H_hist_tmp, Elapsed_tmp + m, h_pred=enc, include_phi=True, add_noise=False)).reshape(H_hist_tmp.size(0), 1)
                    
                    
                    hat_z = Meas_hist_tmp[:,0] - Topo_hist_tmp[:,0]
                    R = 0.001*torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
                    #tmp = torch.linalg.cond(C+R)
                    #if tmp > 10**5:
                    #    print(f'C+R might be ill-conditioned! Cond = {tmp}')
                    
                    out_meas = g.T @ torch.linalg.solve(C+R, hat_z)
                else:
                    out_meas = torch.zeros((1), dtype=torch.float, device=device)

                out = out_topo + out_meas
                z0 = out - tgt
                loss = loss + (c_delta * z0 ** 2)

                count += 1

        return loss, count, h_enc, h_topo
    
    batch_size = batch_size/(nbr_sims-1) #For each step, train over all sims

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    data = {}
    for gidx in range(nbr_sims-1):
        for t in range(n_steps):
            d = dataset[gidx][t]
            tmp = Data(x=d['x'], edge_index=d['edge_index'], edge_attr=d['edge_attr'], node_names=d['node_names'], name_to_idx=d['name_to_idx'], ovls=d['ovls']).to(device)
            data[(gidx, t)] = tmp

    for ep in range(epochs):
        # ---- train ----
        encoder.train(); ftopo.train(); fcov.train()
        train_loss = 0.0
        batch_sample = 1
        loss_batch = torch.zeros((), device=device)
        h_enc = {}
        h_topo = {}
        for t in train_t:
            loss = torch.zeros((), device=device); count = 0
            for gidx in range(nbr_sims-1):
                loss_gidx, count_gidx, h_enc, h_topo = run_t(t, gidx, data, h_enc=h_enc, h_topo=h_topo)
                loss = loss + loss_gidx
                count = count + count_gidx

            step_loss = loss / count if count > 0 else torch.zeros((), dtype=torch.float, device=device)
            loss_batch = loss_batch + step_loss
            train_loss += float((step_loss).detach())

            if batch_sample >= batch_size:
                opt.zero_grad()
                avg_loss = loss_batch/batch_sample
                avg_loss.backward()
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], GRAD_CLIP)
                opt.step()
                batch_sample = 1
                loss_batch = torch.zeros((), device=device)
                h_enc = {}
                h_topo = {}
            else:
                batch_sample = batch_sample + 1

        if batch_sample > 1:
            opt.zero_grad(); 
            avg_loss = loss_batch/batch_sample
            avg_loss.backward()                
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], GRAD_CLIP)
            opt.step()

        train_loss = train_loss / (len(train_t))

        # ---- validate ----
        encoder.eval(); ftopo.eval(); fcov.eval()
        val_loss = 0.0
        h_enc = {}
        h_topo = {}
        with torch.no_grad():
            for t in val_t:
                loss = torch.zeros((), device=device); count = 0
                for gidx in range(nbr_sims-1):
                    loss_gidx, count_gidx, h_enc, h_topo = run_t(t, gidx, data, h_enc=h_enc, h_topo=h_topo)
                    loss = loss + loss_gidx
                    count = count + count_gidx

                val_loss += float((loss / count).detach()) if count > 0 else 0.0

            val_loss = val_loss / (len(val_t))

        scheduler.step(val_loss)
        # ---- early stopping ----
        if val_loss < best_val - min_delta:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = {
                'encoder_state': copy.deepcopy(encoder.state_dict()),
                'ftopo_state': copy.deepcopy(ftopo.state_dict()),
                'fcov_state': copy.deepcopy(fcov.state_dict()),
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep} (best val_loss = {best_val:.6f})")
                break

        msg = f"""
              Epoch: {ep}
              train_loss     = {train_loss}
              val_loss       = {val_loss}
              lr             = {opt.param_groups[0]["lr"]}
              """
        print(msg)

    # restore best weights before save
    if best_state is not None:
        encoder.load_state_dict(best_state['encoder_state'])
        ftopo.load_state_dict(best_state['ftopo_state'])
        fcov.load_state_dict(best_state['fcov_state'])

    # Save models
    model_path = os.path.join(model_dir, 'predictor_models.pth')
    torch.save({'encoder_state': encoder.state_dict(),
                'ftopo_state': ftopo.state_dict(),
                'fcov_state': fcov.state_dict(),
                'mean_delay': mean_delay, 'std_dev_delay': std_dev_delay},
               model_path)
    print('Saved trained models to', model_path)
    return model_path

def calculate_mean_std(train_t, dataset, nbr_sims):
    delay_ms = []
    for gidx in range(nbr_sims):
        for t in train_t:
            curr_overlay = dataset[gidx][t]['ovls']
            for ovl in curr_overlay:
                delay_ms.append(ovl['delay_ms'])

    # Calculate mean
    mean_delay = np.mean(delay_ms, axis=0)
    print(f"Mean of the delay is {mean_delay}")
    
    std_dev_delay = np.std(delay_ms, axis=0)
    print(f"Standard Deviation of the delay is {std_dev_delay}")
    return(mean_delay, std_dev_delay)

def baselines(split_t, dataset, nbr_sims, mu, sd, c_delta):
    mean_tot, pers_tot, n_t = 0.0, 0.0, 0
    for t in split_t[:-1]:
        ms, ps, n = 0.0, 0.0, 0
        for gidx in range(nbr_sims):
            prev = {o['ovl_id']: o['delay_ms'] for o in dataset[gidx][t]['ovls']}
            for o in dataset[gidx][t + 1]['ovls']:
                if o['ovl_id'] in prev:
                    y = o['delay_ms']
                    ms += ((y - mu) / sd) ** 2
                    ps += ((y - prev[o['ovl_id']]) / sd) ** 2
                    n += 1
        if n:
            mean_tot += ms / n; pers_tot += ps / n; n_t += 1
    return c_delta * mean_tot / n_t, c_delta * pers_tot / n_t

if __name__ == '__main__':
    train(epochs=10, batch_size=32)