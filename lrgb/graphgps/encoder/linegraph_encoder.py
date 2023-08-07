import torch
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

from ogb.utils.features import get_bond_feature_dims
from torch_geometric.graphgym.models.encoder import AtomEncoder, BondEncoder


class LineGraphNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, num_classes=None):
        super().__init__()

        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()
        self.bond_embedding_list = torch.nn.ModuleList()
        
        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        node_encoded_features = 0
        # for i in range(batch.x.shape[1]):
        #     encoded_features += self.atom_embedding_list[i](batch.x[:, i])
        for i in range(3):
            node_encoded_features += self.bond_embedding_list[i](batch.x[:, i])
            
        batch.x = node_encoded_features
        return batch


register_node_encoder('AtomLG', LineGraphNodeEncoder)


class LineGraphEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        self.atom_embedding_list = torch.nn.ModuleList()
        self.bond_embedding_list = torch.nn.ModuleList()
        
        for i, dim in enumerate(get_atom_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        for i, dim in enumerate(get_bond_feature_dims()):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, batch):
        edge_encoded_features = 0
        for AtomIdx in range(9):
            edge_encoded_features += self.atom_embedding_list[AtomIdx](batch.edge_attr[:, AtomIdx])
        for edgeIdx in range(9, 12):
            edge_encoded_features += self.bond_embedding_list[edgeIdx-9](batch.edge_attr[:, edgeIdx])
        for edgeIdx in range(12, 15):
            edge_encoded_features += self.bond_embedding_list[edgeIdx-12](batch.edge_attr[:, edgeIdx])

        batch.edge_attr = edge_encoded_features
        return batch


register_edge_encoder('BondLG', LineGraphEdgeEncoder)
