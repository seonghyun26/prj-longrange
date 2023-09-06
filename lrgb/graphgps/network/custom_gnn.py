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
from graphgps.layer.gcn_layer import GCNConvLayer
from graphgps.layer.gat_layer import GATConvLayer
from graphgps.layer.lgnn_layer import graph2linegraph, linegraph2graph, lg2graphNode

from torch_scatter import scatter
from torch_geometric.nn import JumpingKnowledge

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
            if cfg.gnn.lgvariant >= 12:
                print("FLAG - LG dataset2")
                for _ in range(cfg.gnn.layers_mp):
                    layers.append(conv_model(
                        dim_in,
                        dim_in,
                        dropout=cfg.gnn.dropout,
                        residual=cfg.gnn.residual,
                    ))
            else:
                assert("Deprecated version of linegraph")
        else:
            for _ in range(cfg.gnn.layers_mp):
                layers.append(conv_model(
                    dim_in,
                    dim_in,
                    dropout=cfg.gnn.dropout,
                    residual=cfg.gnn.residual,
                ))
        
        if cfg.gnn.linegraph and cfg.gnn.lgvariant >= 20 and cfg.gnn.lgvariant <= 29:
            layers.append(lg2graphNode())
        # if cfg.dataset.task == 'classification':
        #     layers.append(lg2graphNode())
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        
        if cfg.gnn.lgvariant == 13:
            self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner * (cfg.gnn.layers_mp + 1), dim_out=dim_out)
            # self.jump = JumpingKnowledge("cat")
        else:
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
        elif model_type == "gcnconv":
            return GCNConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for idx, module in enumerate(self.children()):
            if self.model_type == 'gcniiconv':
                batch.x0 = batch.x # gcniiconv needs x0 for each layer
                batch = module(batch)
            else:
                if cfg.gnn.lgvariant == 13 and idx == 1 :
                    xJump = [batch.x]
                    for layerIdx, layer in enumerate(module):
                        batch = layer(batch)
                        xJump.append(batch.x)
                    # batch.x = self.jump(xJump)
                    batch.x = torch.cat(xJump, dim=1)
                elif idx != 3:
                    batch = module(batch)
                # if cfg.gnn.lgvariant == 30 and idx == 1:
                #     for layerIdx, layer in enumerate(module):
                #         if layerIdx == 0:
                #             batch = layer(batch)
                #         elif layerIdx == 1:
                #             batch.edge_index_nb = batch.edge_index
                #             batch.edge_attr_nb = batch.edge_attr
                #             batch.edge_index = batch.edge_index_bb
                #             batch.edge_attr = batch.edge_attr_bb
                #             batch = layer(batch)  
                #         elif layerIdx % 2 == 0:
                #             batch.edge_index = batch.edge_index_nb
                #             batch.edge_attr = batch.edge_attr_nb
                #             batch = layer(batch)
                #         else:
                #             batch.edge_index = batch.edge_index_bb
                #             batch.edge_attr = batch.edge_attr_bb
                #             batch = layer(batch)  
        return batch

register_network('custom_gnn', CustomGNN)
