"""
SEGNN
========
Steerable E(3) equivariant neural network
for 3D graphs. The convolutional layer 
uses steerable features and CG products
to perform message passing.

"""

import numpy as np
import torch
from torch_geometric.nn import MessagePassing
from e3nn.nn import BatchNorm
from ._steerable import O3TensorProduct, O3TensorProductSwishGate, Irreps, WeightBalancedIrreps, BalancedIrreps, InstanceNorm
from torch.nn import Module, ModuleList
from torch_geometric.nn import global_add_pool, global_mean_pool


class Conv(MessagePassing):
    """ E(3) equivariant message passing layer. """
    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + hidden_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )

        self.setup_normalisation(norm)

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        """Propagate messages along edges"""
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update note features"""
        input = torch.cat((x, message), dim=-1)
        update = self.update_layer_1(input, node_attr)
        update = self.update_layer_2(update, node_attr)
        x += update  # Residual connection
        return x

class SEGNN(Module):
    """ Steerable E(3) equivariant message passing network """

    def __init__(self, in_dim, out_dim, lmax_attr=1, lmax_h=1, hidden_features=32, num_layers=2, norm=None, pool='avg', init="e3nn"):
        super().__init__()
        # Create network, embedding first
        input_irreps = Irreps(f"{in_dim}x0e")
        edge_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        node_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
        additional_message_irreps = Irreps("1x0e")
        hidden_irreps = WeightBalancedIrreps(Irreps("{}x0e".format(hidden_features)), node_attr_irreps, sh=False, lmax=lmax_h)
        output_irreps = Irreps(f"{out_dim}x0e")

        self.embedding_layer = O3TensorProduct(input_irreps, hidden_irreps, node_attr_irreps)

        # Message passing layers.
        layers = []
        for i in range(num_layers):
            layers.append(
                Conv(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                )
            )
        self.layers = ModuleList(layers)

        # Prepare for output irreps, since the attrs will disappear after pooling
        pooled_irreps = (
            (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
        )
        self.pre_pool1 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )
        self.pre_pool2 = O3TensorProduct(
            hidden_irreps, pooled_irreps, node_attr_irreps
        )
        self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps)
        self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps)
        self.init_pooler(pool)

    def init_pooler(self, pool):
        """Initialise pooling mechanism"""
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def catch_isolated_nodes(self, graph):
        """Isolated nodes should also obtain attributes"""
        if (
            graph.has_isolated_nodes()
            and graph.edge_index.max().item() + 1 != graph.num_nodes
        ):
            nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
            add_attr = graph.node_attr.new_tensor(
                np.zeros((nr_add_attr, graph.node_attr.shape[-1]))
            )
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        graph.node_attr[:, 0] = 1.0


    def forward(self, graph):
        """ SEGNN forward pass """
        x, pos, edge_index, edge_attr, node_attr, batch = graph.atoms, graph.pos, graph.edge_index, graph.edge_attr, graph.node_attr, graph.batch
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None

        self.catch_isolated_nodes(graph)

        # Embed
        x = self.embedding_layer(x.view(-1,1).to(torch.float), node_attr)

        # Pass messages
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, node_attr, batch, additional_message_features)

        # Pre pool
        x = self.pre_pool1(x, node_attr)
        x = self.pre_pool2(x, node_attr)

        # Pool over nodes
        x = self.pooler(x, batch)

        # Predict
        x = self.post_pool1(x)
        x = self.post_pool2(x)
        return x