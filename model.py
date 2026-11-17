# -*- coding: utf-8 -*-
"""TriGAT model file (variant-aware, used by train.py).

The architecture is selected by the environment variable GTE_VARIANT:
    full        (default) 3-layer residual GAT + difference-enhanced edge feature [src;dst;|src-dst|;src*dst]
    noconv3              2-layer residual GAT + difference-enhanced edge feature   (ablation A)
    noedgediff           3-layer residual GAT + plain concatenation [src;dst]      (ablation B)
    twolayer             2-layer residual GAT + plain concatenation [src;dst]      (ResGAT backbone, ablation D/E)
    nofusion             alias of "full"; fusion is disabled at the *training*
                         level by --xx seeds=1,pool=1,snapk=1,ema=0               (ablation C)
    resgat               alias of "twolayer"                                       (ablation E)

When GTE_VARIANT is unset the behaviour is byte-for-byte identical to the
original model.py, so every existing script keeps working.
"""
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
import torch.nn as nn
from arg_parser import parse_args
import torch_geometric.nn as geom_nn


ALIASES = {
    "": "full", "default": "full", "trigat": "full", "tri": "full",
    "nofusion": "full",     # fusion disabled via training flags, not architecture
    "resgat": "twolayer",   # ResGAT backbone = 2-layer residual + plain concat
}
VARIANT = os.environ.get("GTE_VARIANT", "full").strip().lower()
VARIANT = ALIASES.get(VARIANT, VARIANT)
if VARIANT not in ("full", "noconv3", "noedgediff", "twolayer"):
    raise ValueError("unknown GTE_VARIANT: %s" % VARIANT)

USE_CONV3 = VARIANT in ("full", "noedgediff")
USE_DIFF_EDGE = VARIANT in ("full", "noconv3")


class GraphNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256,
                 num_classes=1, heads=8):
        super(GraphNet, self).__init__()
        args = parse_args()
        self.droup_out = args.droup_out

        self.conv1 = GATConv(num_node_features, hidden_channels // heads, heads=heads)
        self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)
        if USE_CONV3:
            self.conv3 = GATConv(hidden_channels, hidden_channels // heads, heads=heads)

        edge_dim = (4 * hidden_channels) if USE_DIFF_EDGE else (2 * hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x, edge_index):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.droup_out, training=self.training)

        x2 = self.conv2(x1, edge_index)

        if USE_CONV3:
            x3 = self.conv3(x2, edge_index)
            x = x1 + x2 + x3          # three-layer residual
        else:
            x = x1 + x2               # two-layer residual

        x = F.relu(x)
        x = F.dropout(x, p=self.droup_out, training=self.training)

        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        if USE_DIFF_EDGE:
            edge_features = torch.cat([src, dst, torch.abs(src - dst), src * dst], dim=-1)
        else:
            edge_features = torch.cat([src, dst], dim=-1)
        edge_prediction = self.mlp(edge_features)

        return edge_prediction.view(-1)
