import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, dropout=0.5):
        super(LinkPredictor, self).__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels*2, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, 1))

        self.dropout = dropout

    def forward(self, x_i, x_j):
        # Concatenate pair of node embeddings
        x = torch.cat([x_i, x_j], dim=-1)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)
