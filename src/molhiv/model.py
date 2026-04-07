import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder

class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, p: float):
        super().__init__()

        self.encoder = AtomEncoder(emb_dim=in_channels)
        self.conv_in = GCNConv(in_channels, hidden_channels, add_self_loops=True, normalize=True)
        self.conv_out = GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True)
        self.dropout = nn.Dropout(p)
        self.relu = nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: int=None):
        out = self.encoder(x)
        out = self.conv_in(out, edge_index)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv_out(out, edge_index)
        out = self.relu(out)
        out = self.dropout(out)
        out = global_mean_pool(out, batch)
        return self.mlp(out)

class GATNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int, dropout: float, add_self_loops: bool):
        super().__init__()

        self.encoder = AtomEncoder(in_channels)
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=dropout, add_self_loops=add_self_loops, concat=True)
        self.conv2 = GATConv(hidden_channels*heads, hidden_channels, heads, dropout=dropout, add_self_loops=add_self_loops, concat=True)
        self.conv3 = GATConv(hidden_channels*heads, hidden_channels, heads, dropout=dropout, add_self_loops=add_self_loops, concat=True)
        self.conv4 = GATConv(hidden_channels*heads, hidden_channels, heads, dropout=dropout, add_self_loops=add_self_loops, concat=True)
        
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout)
        
        self.skip1 = nn.Linear(in_channels, hidden_channels*heads)
        self.skip2 = nn.Linear(hidden_channels*heads, hidden_channels*heads)
        self.skip3 = nn.Linear(hidden_channels*heads, hidden_channels*heads)
        self.skip4 = nn.Linear(hidden_channels*heads, hidden_channels*heads)

        self.bn1 = nn.BatchNorm1d(hidden_channels*heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels*heads)
        self.bn3 = nn.BatchNorm1d(hidden_channels*heads)
        self.bn4 = nn.BatchNorm1d(hidden_channels*heads)

        self.mlp = nn.Linear(hidden_channels*heads, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: int=None):
        x = self.encoder(x)
        x = self.bn1(self.conv1(x, edge_index) + self.skip1(x))
        x = self.relu(x)
        x = self.dropout(x)

        x = self.bn2(self.conv2(x, edge_index) + self.skip2(x))
        x = self.dropout(x)
        x = self.relu(x)

        x = self.bn3(self.conv3(x, edge_index) + self.skip3(x))
        x = self.dropout(x)
        x = self.relu(x)

        x = self.bn4(self.conv4(x, edge_index) + self.skip4(x))
        x = self.dropout(x)
        x = self.relu(x)

        x = global_mean_pool(x, batch)
        return self.mlp(x)
