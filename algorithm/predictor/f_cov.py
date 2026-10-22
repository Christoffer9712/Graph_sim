import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(v):
    v = torch.tensor(float(v))
    return v + torch.log(-torch.expm1(-v))         # stable inverse of softplus, also for large v


class f_cov(nn.Module):
    def __init__(self, in_dim, feat_dim=32, hidden_dim=64, dropout=0.0,
                 n_wp=4, id_weight_init=0.9, ell_t=1/(0.006*8.0), noise_init=0.01):
        """ell_t: time length-scale in the SAME unit as `elapsed` (training passes steps,
        so 1 / (theta * dt) with dt = TIME_STEP in seconds)."""
        super().__init__()
        self.phi = nn.Sequential(                      # learned feature map
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.ell_t = float(ell_t)                      # fixed (O-U: 1/theta), in steps
        self.n_wp = n_wp
        # unconstrained params -> softplus makes them positive
        self.alpha_vec = nn.Parameter(torch.full((n_wp-1,), float(_inv_softplus(id_weight_init))))
        self.raw_noise = nn.Parameter(_inv_softplus(noise_init))   # diagonal R (+ f_topo error)

    def psi(self, h):
        return F.softplus(self.phi(h))

    def noise(self):
        return F.softplus(self.raw_noise)

    def _k_id(self, k_a, k_b, dtype):
        """Sum of alpha_j over the overlay segments (waypoints j -> j+1) that both
        overlays have in common at the same position, i.e. probably the same underlay segment."""
        a = F.softplus(self.alpha_vec).to(dtype)                  # (n_wp-1,) one weight per segment, >= 0
        w_a, w_b = k_a[:, 1:], k_b[:, 1:]                         # waypoints, without the overlay id
        v_a, v_b = w_a >= 0, w_b >= 0                             # real node, not -1 padding
        node_eq = ((w_a[:, None, :] == w_b[None, :, :])           # (A, B, n_wp): same node in slot j
                   & v_a[:, None, :] & v_b[None, :, :])
        seg_eq = (node_eq[..., :-1] & node_eq[..., 1:]).to(dtype)  # (A, B, n_wp-1): both ends of segment j match
        return (seg_eq * a).sum(-1)

    def _k(self, p_a, p_b, lag, k_a=None, k_b=None, include_phi=True):
        K = p_a @ p_b.T
        K = 0*K if not include_phi else K
        if k_a is not None and k_b is not None:
            K = K + self._k_id(k_a.to(p_a.device), k_b.to(p_a.device), p_a.dtype)
        return K * torch.exp(-lag / self.ell_t)

    def forward(self, h_hist, elapsed, h_pred=None, keys_hist=None, keys_pred=None,
                include_phi=True, add_noise=True):
        dtype = self.alpha_vec.dtype
        if h_hist.dim() == 1:
            h_hist = h_hist.unsqueeze(0)                          # (M, N)
        h_hist = h_hist.to(dtype)
        elapsed = elapsed.reshape(-1).to(h_hist)                  # (M,)
        M = h_hist.shape[0]
        assert elapsed.numel() == M, "need one elapsed value per history row"

        p_hist = self.psi(h_hist)                                 # (M, D)

        if h_pred is None:
            lag = (elapsed[:, None] - elapsed[None, :]).abs()     # |t_i - t_j|
            C = self._k(p_hist, p_hist, lag, keys_hist, keys_hist, include_phi=include_phi)  # PSD
            if add_noise:
                C = C + self.noise() * torch.eye(M, dtype=dtype, device=C.device)   # C + R
            return C

        p_pred = self.psi(h_pred.reshape(1, -1).to(h_hist))       # (1, D)
        k_pred = None if keys_pred is None else keys_pred.reshape(1, -1)
        return self._k(p_pred, p_hist, elapsed[None, :], k_pred, keys_hist, include_phi=include_phi).squeeze(0)  # (M,), lag = t* - t_j

    def prior_var(self, h_pred, keys_pred=None):
        p = self.psi(h_pred.reshape(1, -1).to(self.alpha_vec.dtype))
        v = (p * p).sum()
        if keys_pred is not None:
            k = keys_pred.reshape(1, -1)
            v = v + self._k_id(k, k, p.dtype).squeeze()
        return v