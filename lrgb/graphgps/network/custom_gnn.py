import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from graphgps.layer.gcnii_conv_layer import GCN2ConvLayer
from graphgps.layer.gat_conv_layer import GATConvLayer
from graphgps.layer.mlp_layer import MLPLayer
from graphgps.layer.lgnn_layer import graph2linegraph, linegraph2graph

class CustomGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        conv_model = self.build_conv_model(cfg.gnn.layer_type)
        self.model_type = cfg.gnn.layer_type
        layers = []
        if cfg.gnn.linegraph:
            if cfg.gnn.lgvariant == 1:
                layers.append(graph2linegraph(cfg.gnn.lgvariant))
                for _ in range(cfg.gnn.layers_mp):
                    layers.append(conv_model(
                        dim_in*2,
                        dim_in*2,
                        dropout=cfg.gnn.dropout,
                        residual=cfg.gnn.residual,
                    ))
                layers.append(linegraph2graph(cfg.gnn.lgvariant))
            elif cfg.gnn.lgvariant == 2:
                layers.append(graph2linegraph(cfg.gnn.lgvariant))
                for _ in range(cfg.gnn.layers_mp):
                    layers.append(conv_model(
                        dim_in,
                        dim_in,
                        dropout=cfg.gnn.dropout,
                        residual=cfg.gnn.residual,
                    ))
                layers.append(linegraph2graph(cfg.gnn.lgvariant))
            elif cfg.gnn.lgvariant == 3:
                layers.append(graph2linegraph(cfg.gnn.lgvariant))
                for _ in range(cfg.gnn.layers_mp):
                    layers.append(conv_model(
                        dim_in*2,
                        dim_in*2,
                        dropout=cfg.gnn.dropout,
                        residual=cfg.gnn.residual,
                    ))
                layers.append(linegraph2graph(cfg.gnn.lgvariant))
            elif cfg.gnn.lgvariant == 4:
                # line graph readout
                layers.append(graph2linegraph(cfg.gnn.lgvariant))
                for _ in range(cfg.gnn.layers_mp):
                    layers.append(conv_model(
                        dim_in*2,
                        dim_in*2,
                        dropout=cfg.gnn.dropout,
                        residual=cfg.gnn.residual,
                    ))
            elif cfg.gnn.lgvariant == 5:
                layers.append(graph2linegraph(cfg.gnn.lgvariant))
                for _ in range(cfg.gnn.layers_mp):
                    layers.append(conv_model(
                        dim_in*2,
                        dim_in*2,
                        dropout=cfg.gnn.dropout,
                        residual=cfg.gnn.residual,
                    ))
                layers.append(linegraph2graph(cfg.gnn.lgvariant))
            elif cfg.gnn.lgvariant == 6:
                layers.append(graph2linegraph(cfg.gnn.lgvariant))
                for _ in range(cfg.gnn.layers_mp):
                    layers.append(conv_model(
                        dim_in*2,
                        dim_in*2,
                        dropout=cfg.gnn.dropout,
                        residual=cfg.gnn.residual,
                        lgvariant=cfg.gnn.lgvariant
                    ))
                layers.append(linegraph2graph(cfg.gnn.lgvariant))
            else:
                print("wrong cfg in cfg.gnn.lgvariant. Check again")
                assert()
        else:
            for _ in range(cfg.gnn.layers_mp):
                layers.append(conv_model(
                    dim_in,
                    dim_in,
                    dropout=cfg.gnn.dropout,
                    residual=cfg.gnn.residual,
                ))
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'gcniiconv':
            return GCN2ConvLayer
        elif model_type == 'gatconv':
            return GATConvLayer
        elif model_type == 'mlp':
            return MLPLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            if self.model_type == 'gcniiconv':
                batch.x0 = batch.x # gcniiconv needs x0 for each layer
                batch = module(batch)
            else:
                batch = module(batch)
        return batch

register_network('custom_gnn', CustomGNN)
