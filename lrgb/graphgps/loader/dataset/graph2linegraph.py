import torch

def graph2LineGraph(x, edge_index, edge_attr):
    lg_node_attr_edge = edge_attr
    lg_node_attr_node = x[edge_index.T]
    lg_node_attr = torch.cat([lg_node_attr_edge, lg_node_attr_node[:, 0, :], lg_node_attr_node[:, 1, :]], dim=1)
    
    # NOTE: line graph edge index
    lg_node_idx = edge_index.T
    lg_edge_idx_mask = torch.nonzero(
        (lg_node_idx[:, 1, None] == lg_node_idx[:, 0]) &
        (lg_node_idx[:, 0, None] != lg_node_idx[:, 1])
    )
    lg_edge_idx = lg_node_idx[lg_edge_idx_mask]
    
    # NOTE: line graph edge attributes
    lg_edge_attr_node = x[lg_edge_idx[:, 0, 1]]
    edgeStartMask = lg_edge_idx_mask[:, 0].T
    edgeEndMask = lg_edge_idx_mask[:, 1].T
    lg_edge_attr_start = edge_attr[edgeStartMask]
    lg_edge_attr_end = edge_attr[edgeEndMask]
    lg_edge_attr = torch.cat([lg_edge_attr_node, lg_edge_attr_start, lg_edge_attr_end], dim=1)
    
    return lg_node_attr, lg_edge_idx_mask, lg_edge_attr