# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 18:22:22 2023

@author: AmayaGS
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import SAGPooling


class GAT_SAGPool(torch.nn.Module):

    """Graph Attention Network for full slide graph"""

    def __init__(self, dim_in, heads=2, pooling_ratio=0.7):

        super().__init__()

        self.pooling_ratio = pooling_ratio
        self.heads = heads

        self.gat1 = GATv2Conv(dim_in, 512, heads=self.heads, concat=False)
        self.gat2 = GATv2Conv(512, 512, heads=self.heads, concat=False)
        self.gat3 = GATv2Conv(512, 512, heads=self.heads, concat=False)
        self.gat4 = GATv2Conv(512, 512, heads=self.heads, concat=False)

        self.topk1 = SAGPooling(512, pooling_ratio)
        self.topk2 = SAGPooling(512, pooling_ratio)
        self.topk3 = SAGPooling(512, pooling_ratio)
        self.topk4 = SAGPooling(512, pooling_ratio)

        self.lin1 = torch.nn.Linear(512 * 2, 512)
        self.lin2 = torch.nn.Linear(512, 512 // 2)
        self.lin3 = torch.nn.Linear(512 // 2, 5)


    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        x, edge_index, _, batch, _, _= self.topk1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat2(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        x, edge_index, _, batch, _, _= self.topk2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat3(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        x, edge_index, _, batch, _, _= self.topk3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.gat4(x, edge_index)
        x = F.relu(x)
        #x = F.dropout(x, p=0.1, training=self.training)
        x, edge_index, _, batch, _, _= self.topk4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x_logits = self.lin3(x)
        x_out = F.softmax(x_logits, dim=1)

        return x_logits, x_out