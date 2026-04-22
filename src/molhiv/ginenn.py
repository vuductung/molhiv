import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class GINENN(nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int, p: float, out_channels: int):
        super().__init__()

        self.atom_encoder = AtomEncoder(hidden_channels)
        self.bond_encoder = BondEncoder(hidden_channels)
        self.hidden_channels = hidden_channels
        self.p = p

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = self.make_gin_nn_module()
            conv = GINEConv(mlp, train_eps=True, edge_dim=hidden_channels)
            bn = nn.BatchNorm1d(hidden_channels)
            self.convs.append(conv)
            self.bns.append(bn)

        self.dropout = nn.Dropout(p)
        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_attr: torch.Tensor):

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        
        for conv, bn in zip(self.convs, self.bns):
            x = (conv(x, edge_index, edge_attr) + x).relu()
            x = bn(x)
            x = self.dropout(x)

        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.classifier(x)

    def make_gin_nn_module(self):
        return nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.BatchNorm1d(self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.p)
        )