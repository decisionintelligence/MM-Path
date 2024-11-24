import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        adj = self.build_sparse_adj_matrix(edge_index, num_nodes).coalesce()
        deg = self.calculate_degree(adj)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_adj = self.normalize_adj(adj, deg_inv_sqrt).coalesce().to(x.device)
        x = self.linear(x)
        out = torch.sparse.mm(norm_adj, x)
        return out

    def build_sparse_adj_matrix(self, edge_index, num_nodes):
        indices = edge_index
        values = torch.ones(indices.size(1), dtype=torch.float32, device=indices.device)
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=indices.device)
        return adj

    def add_self_loops(self, adj, num_nodes):
        indices = torch.arange(0, num_nodes, dtype=torch.long, device=adj.device)
        indices = torch.stack([indices, indices], dim=0)
        values = torch.ones(num_nodes, dtype=torch.float32, device=adj.device)
        self_loop_adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=adj.device)
        adj = adj + self_loop_adj
        return adj

    def calculate_degree(self, adj):
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        return deg

    def normalize_adj(self, adj, deg_inv_sqrt):
        row, col = adj.indices()
        values = deg_inv_sqrt[row] * adj.values() * deg_inv_sqrt[col]
        norm_adj = torch.sparse_coo_tensor(adj.indices(), values, adj.size(), device=adj.device)
        return norm_adj

def merge_graphs(x, edge_index):
    batch_size, num_nodes, num_features = x.shape
    _, _, num_edges = edge_index.shape
    merged_x = x.reshape(batch_size * num_nodes, num_features)
    row, col = edge_index[:, 0, :], edge_index[:, 1, :]
    row_offset = torch.arange(batch_size, device=edge_index.device).repeat_interleave(num_edges) * num_nodes
    merged_row = (row.reshape(-1) + row_offset).reshape(-1)
    merged_col = (col.reshape(-1) + row_offset).reshape(-1)
    merged_edge_index = torch.stack([merged_row, merged_col], dim=0)
    return merged_x, merged_edge_index

def unmerge_graphs(merged_x, batch_size, num_nodes, num_features):
    return merged_x.reshape(batch_size, num_nodes, num_features)

class BatchGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(BatchGCN, self).__init__()
        self.out_channels = out_channels
        self.conv1 = GCNConv(64, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)

    def forward(self, x, edge_index):
        edge_index = edge_index.transpose(1, 2)
        batch_size, num_nodes, _ = x.shape
        merged_x, merged_edge_index = merge_graphs(x, edge_index)
        merged_x = self.conv1(merged_x, merged_edge_index)
        merged_x = F.relu(merged_x)
        merged_x = self.conv2(merged_x, merged_edge_index)
        merged_x = F.relu(merged_x)
        merged_x = self.conv3(merged_x, merged_edge_index)
        x = unmerge_graphs(merged_x, batch_size, num_nodes, self.out_channels)
        return x
