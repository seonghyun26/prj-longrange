import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_scatter import scatter, scatter_mean

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.config import cfg

   
class graph2linegraph(nn.Module):
    # Convert graph into a line graph
    # NOTE: x, x0, edge_index, edge_attr
    def __init__(self, variant=1):
        super().__init__()
        self.variant = variant
        
    def forward(self, batch):
        # Save original information in batch
        batch.shape = batch.x.shape
        batch.org_x = batch.x
        batch.org_edge_index = batch.edge_index
        batch.org_edge_attr = batch.edge_attr
        batch.org_batch = batch.batch
        if hasattr(batch, 'x0'):
            batch.org_x0 = batch.x0
        
        # Add bb edges
        if self.variant == 8:
            # print("FLAG - v8")
            node_count = torch.unique(batch.edge_index[0], return_counts=True)
            leaf_node = torch.where(node_count[1] == 1)[0]
            bb_edge = torch.stack((leaf_node, leaf_node), dim=0)
            batch.edge_index = torch.cat((batch.edge_index, bb_edge), dim=1)
            batch.edge_attr = torch.cat([batch.edge_attr, torch.zeros(leaf_node.shape[0], batch.edge_attr.shape[1], device=batch.y.device)], dim=0)
            indices = torch.argsort(batch.edge_index.T, dim=0)
            batch.edge_index = batch.edge_index.T[indices[:,0], :].T
            batch.edge_attr = batch.edge_attr[indices[:,0], :]
            
        
        lg_node_idx = batch.edge_index.T
        batch.lg_node_idx = lg_node_idx
        
        # NOTE: Line graph node feature 
        lg_x = batch.x[lg_node_idx]
        batch.x = torch.reshape(lg_x, (lg_x.shape[0], -1))
        lg_x = None
        if hasattr(batch, 'edge_attr'):
            batch.x = torch.stack([batch.x, batch.edge_attr.repeat(1,2)], dim=2).mean(dim=2)
            
        if hasattr(batch, 'x0'):
            lg_x0 = batch.x0[lg_node_idx]
            batch.x0 = torch.reshape(lg_x0, (lg_x0.shape[0], -1))
            lg_x0 = None


        # NOTE: line graph edge index
        # startNode, endNode = lg_node_idx[:, 0], lg_node_idx[:, 1]   
        lg_edge_idx = torch.nonzero((lg_node_idx[:, 1, None] == lg_node_idx[:, 0]) & (lg_node_idx[:, 0, None] != lg_node_idx[:, 1]))
        batch.edge_index = lg_edge_idx.T
        
        
        # NOTE: line graph edge attribute
        new_edge_idx = lg_node_idx[lg_edge_idx]
        lg_edge_attr = batch.org_x[new_edge_idx[:, 0, 1]].repeat(1,2)
        if hasattr(batch, 'edge_attr'):
            startEdge = new_edge_idx[:, 0]
            endEdge = new_edge_idx[:, 1]
            startIndices = torch.where(torch.all(batch.org_edge_index.T[:, None] == startEdge, dim=2))
            startEdgeAttr = batch.org_edge_attr[scatter(startIndices[0], startIndices[1])]
            endIndices = torch.where(torch.all(batch.org_edge_index.T[:, None] == endEdge, dim=2))
            endEdgeAttr = batch.org_edge_attr[scatter(endIndices[0], endIndices[1], dim=0)]
            if self.variant == 8 and startEdgeAttr.shape[0] < endEdgeAttr.shape[0]:
                startEdgeAttr = torch.cat([startEdgeAttr, torch.zeros(endEdgeAttr.shape[0] - startEdgeAttr.shape[0], startEdgeAttr.shape[1], device=batch.y.device)], dim=0)
            elif self.variant == 8 and startEdgeAttr.shape[0] > endEdgeAttr.shape[0]:
                endEdgeAttr = torch.cat([endEdgeAttr, torch.zeros(startEdgeAttr.shape[0] - endEdgeAttr.shape[0], endEdgeAttr.shape[1], device=batch.y.device)], dim=0)
            temp = torch.cat([startEdgeAttr, endEdgeAttr], dim=1)
            lg_edge_attr = torch.stack([lg_edge_attr, temp], dim=2).mean(dim=2)
            del startIndices
            del endIndices
            del startEdgeAttr
            del endEdgeAttr 
        batch.edge_attr = lg_edge_attr
        del lg_edge_idx
        
        if self.variant ==7:
            ptr = batch.ptr
            batch_size = batch.y.shape[0]
            new_batch = torch.where((batch.org_edge_index [0] >= ptr[:-1].unsqueeze(1)) & (batch.org_edge_index [0] < ptr[1:].unsqueeze(1)))
            batch.batch = new_batch[0]

        return batch

class linegraph2graph(nn.Module):
    # Convert line graph to graph
    # NOTE: x, x0, edge_index, edge_attr
    def __init__(self, variant=1):
        super().__init__()
        self.variant = variant
        
    def pad(self, tensor, originalShape):
        if originalShape[0] - tensor.shape[0] > 0:
            return torch.cat([tensor, torch.zeros(originalShape[0] - tensor.shape[0], originalShape[1], device=tensor.device)])
        else:
            return tensor
    
    def forward(self, batch):
        # Recover node feature
        shape = batch.shape
        frontNode = scatter_mean(batch.x, batch.lg_node_idx[:,0], dim=0)[:, shape[1]:]
        frontNode = self.pad(frontNode, shape)
        backNode = scatter_mean(batch.x, batch.lg_node_idx[:,1], dim=0)[:, :shape[1]]
        backNode = self.pad(backNode, shape)
        batch.x = torch.add(
            frontNode,
            backNode
        )
        
        # Recover edge feature
        shape = batch.org_edge_attr.shape
        frontEdge = scatter_mean(batch.edge_attr, batch.edge_index.T[:,0], dim=0)[:, shape[1]:]
        frontEdge = self.pad(frontEdge, shape)
        backEdge = scatter_mean(batch.edge_attr, batch.edge_index.T[:,1], dim=0)[:, :shape[1]]
        backEdge = self.pad(backEdge, shape)
        if self.variant == 8 and frontEdge.shape[0] > backEdge.shape[0]:
            backEdge = torch.cat([backEdge, torch.zeros(frontEdge.shape[0] - backEdge.shape[0], backEdge.shape[1], device=batch.y.device)], dim=0)
        elif self.variant == 8 and frontEdge.shape[0] < backEdge.shape[0]:
            frontEdge = torch.cat([frontEdge, torch.zeros(backEdge.shape[0] - frontEdge.shape[0], frontEdge.shape[1], device=batch.y.device)], dim=0)
        batch.edge_attr = torch.add(
            frontEdge,
            backEdge,
        )
        
        # Garbage Collecting 
        del frontNode
        del backNode
        del frontEdge
        del backEdge
        del shape
        
        if hasattr(batch, 'x0'):
            batch.x0 = batch.org_x0
        batch.edge_index = batch.org_edge_index
        
        return batch