import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.data import Batch
from config import MAX_ISL_LINK_DISTANCE

NODE_TYPES = ["aircraft", "satellite", "gateway", "aviation", "upf"]
_TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}

def snapshot_to_pyg(
    H,
    node_attrs=("position", "velocity"),      # which attrs to keep
    edge_attrs=("distance",),
    include_type_onehot=True,
    pos_scale=[50, 50, 600_000],              # Typical Lat/Lon deg, altitude 600 km
    vel_scale = [0.1, 0.1],                    # Lat/Lon deg/s              
    dist_scale = MAX_ISL_LINK_DISTANCE/2,
):
    """
    Convert one snapshot to a PyG Data object, selecting only the listed
    attributes. Node ordering is fixed by sorted node name so that embeddings
    are comparable across timesteps.
    """
    nodes = sorted(H.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    # ---- node features ---------------------------------------------------
    rows = []
    for n in nodes:
        d = H.nodes[n]
        feat = []
        for a in node_attrs:
            v = d[a]
            if a == "position":                     # (dx, dy) tuple
                feat.extend([v[0] / pos_scale[0], v[1] / pos_scale[1], v[2] / pos_scale[2]])
            elif a == "velocity":
                feat.extend([v[0] / vel_scale[0], v[1] / vel_scale[1]])
            elif isinstance(v, (tuple, list, np.ndarray)):
                feat.extend(np.asarray(v, dtype=float).ravel())
            else:
                feat.append(float(v))
        if include_type_onehot:
            oh = [0.0] * len(NODE_TYPES)
            oh[_TYPE_IDX[d["node_type"]]] = 1.0
            feat.extend(oh)
        rows.append(feat)
    x = torch.tensor(rows, dtype=torch.float32)

    # ---- edges: undirected -> both directions ----------------------------
    src, dst, eattr = [], [], []
    for u, v, ed in H.edges(data=True):
        ef = []
        for a in edge_attrs:
            val = ed[a]
            ef.append(val / dist_scale if a == "distance" else float(val))
        for a_, b_ in ((u, v), (v, u)):
            src.append(idx[a_])
            dst.append(idx[b_])
            eattr.append(ef)

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(eattr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(edge_attrs)), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_names = nodes          # keep the mapping for later lookup
    data.name_to_idx = idx
    return data

def overlay_dist(data, overlay):
    dist = 0.0
    for node_src, node_dst in zip(overlay[:-1], overlay[1:]):
        src_pos = data.x[data.name_to_idx[node_src]][0:2]
        dst_pos = data.x[data.name_to_idx[node_dst]][0:2]
        dist = dist + torch.norm(src_pos - dst_pos)
    return dist

def _add_overlay_nodes_pyg(data, overlays):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    name_to_idx = dict(data.name_to_idx)
    node_names = list(data.node_names)

    virtual, src_virtual, dst_node, src_node, dst_virtual = [], [], [], [], []
    for i, overlay in enumerate(overlays):
        ovl_idx = len(virtual) #x.size(0) + len(virtual)
        name_to_idx[f"OVL-{i}"] = ovl_idx
        node_names.append(f"OVL-{i}")
        virtual.append([0])#[overlay_dist(data, overlay['ovl_path'])])
        for node in overlay["ovl_path"]:
            tgt = name_to_idx[node]
            src_virtual += [ovl_idx]
            src_node += [tgt]
            dst_virtual += [ovl_idx]
            dst_node += [tgt]

    hetero = HeteroData()
    hetero['node'].x = x
    hetero['virtual'].x = torch.tensor(virtual, dtype=torch.float32)

    hetero['node', 'n-v', 'virtual'].edge_index = torch.tensor((src_node, dst_virtual), dtype=torch.long)
    hetero['virtual', 'v-n', 'node'].edge_index = torch.tensor((src_virtual, dst_node), dtype=torch.long)
    hetero['node', 'n-n', 'node'].edge_index = edge_index
    hetero['node', 'n-n', 'node'].edge_attr = edge_attr
    
    
    #out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    hetero.node_names = node_names
    hetero.name_to_idx = name_to_idx
    return hetero

class GraphEncoder:
    def __init__(self, encoder, device="cpu"):
        self.encoder = encoder
        self.device = device

    def encode_overlays_pyg_batched(self, data, overlays):
        graphs = []
        for overlay in overlays:
            g = _add_overlay_nodes_pyg(data, [overlay])  # still single-overlay
            graphs.append(g.to(self.device))

        # Disjoint batch: no edges between sub-graphs, so overlay i's virtual
        # node only attends within overlay i's own topology copy.
        batch = Batch.from_data_list(graphs)

        encoding = self.encoder.forward(
            batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict
        )

        # Every sub-graph had exactly one virtual node, so row i == overlay i.
        return {overlay['ovl_id']: encoding['virtual'][i]
                for i, overlay in enumerate(overlays)}