"""
GNN + DQN tunnel selection agent.

Architecture
────────────
GNNEncoder   : 2-layer message-passing GNN over the full topology graph.
               Produces a HIDDEN_DIM embedding per node.
DQNHead      : scores a (state, tunnel_action) pair with a scalar Q-value.
               state  = aircraft_embedding ‖ demand_features   (HIDDEN_DIM + 3)
               action = linkType_onehot ‖ firstHop_emb ‖ upf_emb
                                                        (2 + 2·HIDDEN_DIM)
TunnelAgent  : wraps the two networks, an experience replay buffer, and
               epsilon-greedy exploration.

Training loop
─────────────
  1. select_tunnels() — called each timestep; epsilon-greedy action choice.
  2. store_reward()   — called after sendData(); pushes transition to buffer.
  3. train_step()     — samples a minibatch and updates the online DQN.
  4. Target network is soft-copied every TARGET_UPDATE steps.
"""

import random
from collections import deque

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import LinkType, TrafficDescription, TunnelDescription
from ground_network.nodes import GroundNodeType

# ─── Hyper-parameters ────────────────────────────────────────────────────────

HIDDEN_DIM    = 64
GNN_LAYERS    = 2
DEMAND_DIM    = 3              # normalised fiveQI, normalised BW, UPF index
LINK_TYPE_DIM = 2              # one-hot for LinkType.SA2A / DA2G
STATE_DIM     = HIDDEN_DIM + DEMAND_DIM
ACTION_DIM    = LINK_TYPE_DIM + 2 * HIDDEN_DIM

GAMMA         = 0.99
LR            = 1e-3
BUFFER_CAP    = 10_000
BATCH_SIZE    = 64
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 0.995          # multiplicative decay per train_step call
TARGET_UPDATE = 20             # steps between target network copies
MAX_BW        = 200.0          # Mbps — used to normalise BW demand feature

# ─── Node / edge vocabulary ───────────────────────────────────────────────────

_NODE_IDX = {
    'aircraft':              0,
    GroundNodeType.GATEWAY:  1,
    GroundNodeType.AVIATION: 2,
    GroundNodeType.UPF:      3,
    'satellite':             4,
}
N_NODE_TYPES = 5

_EDGE_IDX = {
    'isl':               0,
    'gw_sat':            1,
    'aviation_aviation': 2,
    'aviation_gateway':  3,
    'upf_gw':            4,
    'upf_av':            5,
    'a2s':               6,
    'a2g':               7,
    'ground_grid':       8,
}
N_EDGE_TYPES = 9

# ─── Graph → tensors ─────────────────────────────────────────────────────────

def _graph_to_tensors(
    graph: nx.Graph,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Convert a NetworkX graph to the tensors the GNN needs.

    Returns
    -------
    node_feats  : (N, N_NODE_TYPES)   — one-hot node type
    edge_index  : (2, 2E)             — directed (both directions per edge)
    edge_feats  : (2E, N_EDGE_TYPES)  — one-hot edge type
    node_idx    : {node_id: row_index}
    """
    nodes    = list(graph.nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    N        = len(nodes)

    node_feats = torch.zeros(N, N_NODE_TYPES)
    for n, d in graph.nodes(data=True):
        nt = d.get('node_type', 'satellite')
        node_feats[node_idx[n], _NODE_IDX.get(nt, 4)] = 1.0

    src, dst, efeats = [], [], []
    for u, v, d in graph.edges(data=True):
        lt  = d.get('link_type', 'isl')
        ef  = torch.zeros(N_EDGE_TYPES)
        ef[_EDGE_IDX.get(lt, 0)] = 1.0
        src += [node_idx[u], node_idx[v]]
        dst += [node_idx[v], node_idx[u]]
        efeats += [ef, ef]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_feats = torch.stack(efeats)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_feats = torch.zeros(0, N_EDGE_TYPES)

    return node_feats, edge_index, edge_feats, node_idx

# ─── GNN ─────────────────────────────────────────────────────────────────────

class _GNNLayer(nn.Module):
    """One message-passing step with mean aggregation and a residual connection."""

    def __init__(self, dim: int, edge_dim: int):
        super().__init__()
        self.msg = nn.Linear(dim + edge_dim, dim)
        self.upd = nn.Sequential(nn.Linear(dim + dim, dim), nn.ReLU())
        self._dim = dim

    def forward(
        self,
        x:          torch.Tensor,   # (N, dim)
        edge_index: torch.Tensor,   # (2, E)
        edge_feats: torch.Tensor,   # (E, edge_dim)
    ) -> torch.Tensor:
        N = x.size(0)
        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            msgs = F.relu(self.msg(torch.cat([x[src], edge_feats], dim=-1)))
            agg  = torch.zeros(N, self._dim, device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msgs), msgs)
            deg  = torch.bincount(dst, minlength=N).float().clamp(min=1).unsqueeze(-1)
            agg  = agg / deg
        else:
            agg = torch.zeros(N, self._dim, device=x.device)
        return self.upd(torch.cat([x, agg], dim=-1))


class GNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj   = nn.Linear(N_NODE_TYPES, HIDDEN_DIM)
        self.layers = nn.ModuleList(
            [_GNNLayer(HIDDEN_DIM, N_EDGE_TYPES) for _ in range(GNN_LAYERS)]
        )

    def forward(self, nf, ei, ef) -> torch.Tensor:
        x = F.relu(self.proj(nf))
        for layer in self.layers:
            x = x + layer(x, ei, ef)   # residual keeps gradients healthy
        return x                        # (N, HIDDEN_DIM)

# ─── DQN head ────────────────────────────────────────────────────────────────

class DQNHead(nn.Module):
    """
    Q(state, action) → scalar.
    We evaluate one tunnel at a time so the action space can vary per step.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64),                     nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)

# ─── Replay buffer ───────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_CAP):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, n: int = BATCH_SIZE):
        return random.sample(self._buf, n)

    def __len__(self):
        return len(self._buf)

# ─── Agent ───────────────────────────────────────────────────────────────────

class TunnelAgent:
    def __init__(self):
        self.gnn    = GNNEncoder()
        self.dqn    = DQNHead()
        self.target = DQNHead()
        self.target.load_state_dict(self.dqn.state_dict())
        self.buffer = ReplayBuffer()
        self.opt    = torch.optim.Adam(
            list(self.gnn.parameters()) + list(self.dqn.parameters()), lr=LR
        )
        self.epsilon  = EPS_START
        self._step    = 0
        # Stash (state, action) vectors between select_tunnels and store_reward
        self._pending: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    # ── graph encoding ────────────────────────────────────────────────────────

    def _encode(self, graph: nx.Graph):
        nf, ei, ef, nidx = _graph_to_tensors(graph)
        with torch.no_grad():
            emb = self.gnn(nf, ei, ef)
        return emb, nidx

    # ── feature builders ──────────────────────────────────────────────────────

    def _demand_vec(
        self, demand: TrafficDescription, upf_nodes: list
    ) -> torch.Tensor:
        """3-dim vector: normalised fiveQI, normalised BW, normalised UPF index."""
        upf_sorted = sorted(upf_nodes)
        upf_idx    = upf_sorted.index(demand.UPF) / max(len(upf_sorted) - 1, 1)
        return torch.tensor(
            [demand.fiveQI / 9.0, demand.BW / MAX_BW, upf_idx], dtype=torch.float32
        )

    def _action_vec(
        self, opt: dict, emb: torch.Tensor, nidx: dict
    ) -> torch.Tensor:
        """
        Encode a structural tunnel option as a fixed-size vector:
          linkType one-hot (2) | firstHop embedding | UPF embedding
        """
        lt_oh = torch.zeros(LINK_TYPE_DIM)
        lt_oh[int(opt['linkType']) - 1] = 1.0

        def _node_emb(nid):
            i = nidx.get(nid)
            return emb[i] if i is not None else torch.zeros(HIDDEN_DIM)

        return torch.cat([lt_oh, _node_emb(opt['firstHop']), _node_emb(opt['UPF'])])

    # ── tunnel enumeration ────────────────────────────────────────────────────

    def _enumerate_options(
        self, graph: nx.Graph, aircraft_id: str, target_upf: str
    ) -> list[dict]:
        """
        Find all structurally valid (linkType, firstHop, GW, UPF) tuples.

        Uses a single BFS per candidate first hop to avoid O(N²) nx.has_path calls.
        Limits to one GW per satellite to keep the action space tractable.
        """
        if aircraft_id not in graph:
            return []

        nbrs    = set(graph.neighbors(aircraft_id))
        gw_set  = {n for n, d in graph.nodes(data=True)
                   if d.get('node_type') == GroundNodeType.GATEWAY}
        options = []

        # SA2A: satellite neighbours
        sat_nbrs = [
            n for n in nbrs
            if graph.nodes[n].get('node_type') not in
               {GroundNodeType.GATEWAY, GroundNodeType.AVIATION,
                GroundNodeType.UPF, 'aircraft'}
        ]
        for sat in sat_nbrs:
            reachable = set(nx.single_source_shortest_path_length(graph, sat))
            if target_upf not in reachable:
                continue
            # Pick the first reachable GW (shortest-hop, stable sort)
            gw = next((g for g in sorted(gw_set) if g in reachable), None)
            if gw:
                options.append({
                    'linkType': LinkType.SA2A,
                    'firstHop': sat, 'GW': gw, 'UPF': target_upf,
                })

        # DA2G: aviation node neighbours
        av_nbrs = [
            n for n in nbrs
            if graph.nodes[n].get('node_type') == GroundNodeType.AVIATION
        ]
        for av in av_nbrs:
            reachable = set(nx.single_source_shortest_path_length(graph, av))
            if target_upf in reachable:
                options.append({
                    'linkType': LinkType.DA2G,
                    'firstHop': av, 'GW': '', 'UPF': target_upf,
                })

        return options

    # ── public API ────────────────────────────────────────────────────────────

    def select_tunnels(
        self,
        demands:     list[TrafficDescription],
        graph:       nx.Graph,
        aircraft_id: str,
    ) -> list[TunnelDescription]:
        """Epsilon-greedy tunnel selection for all current demands."""
        emb, nidx   = self._encode(graph)
        upf_nodes   = [n for n, d in graph.nodes(data=True)
                       if d.get('node_type') == GroundNodeType.UPF]
        ac_i        = nidx.get(aircraft_id)
        ac_emb      = emb[ac_i] if ac_i is not None else torch.zeros(HIDDEN_DIM)
        tunnels     = []

        for demand in demands:
            options = self._enumerate_options(graph, aircraft_id, demand.UPF)
            if not options:
                continue

            state = torch.cat([ac_emb, self._demand_vec(demand, upf_nodes)])

            if random.random() < self.epsilon:
                chosen = random.choice(options)
            else:
                with torch.no_grad():
                    q_vals = torch.stack([
                        self.dqn(
                            state.unsqueeze(0),
                            self._action_vec(opt, emb, nidx).unsqueeze(0),
                        )
                        for opt in options
                    ])
                chosen = options[q_vals.argmax().item()]

            key = (demand.fiveQI, demand.BW, demand.UPF)
            self._pending[key] = (state, self._action_vec(chosen, emb, nidx))

            tunnels.append(TunnelDescription(
                fiveQI=demand.fiveQI,
                BW=demand.BW,
                linkType=chosen['linkType'],
                firstHop=chosen['firstHop'],
                GW=chosen['GW'],
                UPF=chosen['UPF'],
            ))

        return tunnels

    def store_reward(
        self,
        demands:      list[TrafficDescription],
        per_list:     list[float],
        latency_list: list[float],
    ) -> None:
        """
        Compute rewards and push transitions to the replay buffer.
        Reward design: penalise latency for delay-sensitive QoS classes (low fiveQI),
        penalise PER for all classes.  Latency is converted to milliseconds.
        """
        for demand, per, latency in zip(demands, per_list, latency_list):
            key = (demand.fiveQI, demand.BW, demand.UPF)
            if key not in self._pending:
                continue
            state, action = self._pending.pop(key)

            if latency == float('inf') or per >= 1.0:
                reward = -100.0   # hard penalty for no-path
            else:
                lat_weight = 2.0 if demand.fiveQI <= 4 else 0.5
                reward = -(lat_weight * latency * 1000.0 + 10.0 * per)

            # We treat each timestep as a terminal episode for simplicity;
            # a proper multi-step return can be added once the simulator is stable.
            self.buffer.push(state, action, reward,
                             torch.zeros_like(state), done=True)

    def train_step(self) -> float | None:
        """
        One gradient step on the online DQN.
        Returns the loss value, or None if the buffer is not yet large enough.
        """
        if len(self.buffer) < BATCH_SIZE:
            return None

        batch       = self.buffer.sample(BATCH_SIZE)
        states      = torch.stack([b[0] for b in batch])
        actions     = torch.stack([b[1] for b in batch])
        rewards     = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.stack([b[3] for b in batch])
        dones       = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        q_pred  = self.dqn(states, actions)
        with torch.no_grad():
            q_next  = self.target(next_states, actions)
            q_target = rewards + GAMMA * (1.0 - dones) * q_next

        loss = F.mse_loss(q_pred, q_target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.gnn.parameters()) + list(self.dqn.parameters()), max_norm=1.0
        )
        self.opt.step()

        self.epsilon  = max(EPS_END, self.epsilon * EPS_DECAY)
        self._step   += 1
        if self._step % TARGET_UPDATE == 0:
            self.target.load_state_dict(self.dqn.state_dict())

        return loss.item()


# ─── Module-level singleton ───────────────────────────────────────────────────

_agent: TunnelAgent | None = None


def get_agent() -> TunnelAgent:
    global _agent
    if _agent is None:
        _agent = TunnelAgent()
    return _agent


def get_tunnels(
    traffic_demand: list[TrafficDescription],
    dt_tunnel:      float,
    graph:          nx.Graph,
    aircraft_id:    str = 'aircraft',
) -> list[TunnelDescription]:
    """Entry point called by Aircraft.setUpTunnels."""
    return get_agent().select_tunnels(traffic_demand, graph, aircraft_id)