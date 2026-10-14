import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

RELATIONS = [
    ("node", "n-v", "virtual"),
    ("virtual", "v-n", "node"),
    ("node", "n-n", "node")
]

class HeteroGATv2Encoder(nn.Module):
    def __init__(self, node_in, virtual_in, hidden_dim=128, out_dim=64,
                 heads=2, n_layers=3, dropout=0.1):
        assert(hidden_dim % heads == 0)
        super().__init__()
        self.n_layers = n_layers
        self.dropout = dropout

        # project both node types to a common width first
        self.lin_node = nn.Linear(node_in, hidden_dim)
        self.lin_virtual = nn.Linear(virtual_in, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [hidden_dim] * n_layers + [out_dim]
        for l in range(n_layers):
            d_in, d_out = dims[l], dims[l + 1]
            last = l == n_layers - 1
            per_head = d_out if last else d_out // heads
            conv = HeteroConv(
                {
                    ("node", "n-v", "virtual"): GATv2Conv(d_in, per_head, heads=heads, edge_dim=None,
                                                concat=not last, add_self_loops=True, dropout=dropout, 
                                                share_weights=False, residual=True),
                    ("virtual", "v-n", "node"): GATv2Conv(d_in, per_head, heads=heads, edge_dim=None,
                                                concat=not last, add_self_loops=True, dropout=dropout,
                                                share_weights=False, residual=True),
                    ("node", "n-n", "node"): GATv2Conv(d_in, per_head, heads=heads, edge_dim=1,
                                                concat=not last, add_self_loops=True, dropout=dropout, 
                                                share_weights=False, residual=True)
                },
                aggr="sum",
            )
            self.convs.append(conv)
            self.norms.append(nn.ModuleDict({
                "node": nn.LayerNorm(d_out),
                "virtual": nn.LayerNorm(d_out),
            }))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x = {"node": self.lin_node(x_dict["node"]),
             "virtual": self.lin_virtual(x_dict["virtual"])}

        for l in range(self.n_layers):
            last = l == self.n_layers - 1
            x = self.convs[l](x, edge_index_dict, edge_attr_dict=edge_attr_dict)
            
            new_x = {}
            for k, v in x.items():
                v = self.norms[l][k](v)
                if not last:
                    v = F.dropout(F.elu(v), p=self.dropout, training=self.training)
                new_x[k] = v
            x = new_x
        return x