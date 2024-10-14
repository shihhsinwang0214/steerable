from math import pi as PI

import torch
from torch import Tensor
from torch.nn import Linear, Sequential

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models.schnet import GaussianSmearing, RadiusInteractionGraph, ShiftedSoftplus

import numpy as np
import torch
from torch.nn import Embedding, Linear, Module, ModuleList
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.graphgym import global_add_pool
from torch_geometric.nn import Sequential

class SCHNET(Module):
    def __init__(self, hidden_channels, out_channels, hidden_layers, radius, num_gaussians, num_filters, max_num_neighbors):
        super(SCHNET, self).__init__()
        self.hidden_layers = hidden_layers
        self.num_gaussians = num_gaussians
        self.num_filters = num_filters
        self.radius = radius

        self.distance_expansion =  GaussianSmearing(0.0, radius, num_gaussians)
        self.interaction_graph = RadiusInteractionGraph(radius, max_num_neighbors)
        # conv
        self.layers = ModuleList([])
        for _ in range(hidden_layers): #Interaction Blocks
            self.layers.append(Sequential('x, edge_index, edge_weight, edge_attr',[
                (CFConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_gaussians=self.num_gaussians,
                    num_filters=self.num_filters,
                    cutoff=self.radius,
                ), 'x, edge_index, edge_weight, edge_attr -> x1'),
                (ShiftedSoftplus(), 'x1 -> x2'),
                (Linear(hidden_channels, hidden_channels), 'x2 -> x3'),

            ]))
        # decoder
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, out_channels)

    def forward(self, x, pos, edge_index, batch):

        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for seq in self.layers:
            x = seq(x, edge_index, edge_weight, edge_attr)

        # decoder
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        x = global_add_pool(x, batch)
        return x.squeeze(1)



class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_gaussians: int,
        num_filters: int,
        cutoff: float,
    ):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = torch.nn.Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor) -> Tensor:

        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W
