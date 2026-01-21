# graph.py

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple
import torch

def build_normalized_adj(
    num_users: int,
    num_items: int,
    train_interactions: torch.Tensor,
    device : torch.device
) -> torch.Tensor:
    """
    Build symmetric normalized adjacency matrix for LightGCN:
        A_hat = D^{-1/2} A D^{-1/2}
    where A is the bipartite adjacency constructed from train_interactions.

    Args:
        num_users: number of users
        num_items: number of items
        train_interactions: LongTensor [N_train, 2] with (user_id, item_id)
        device: torch.device
    Returns:
        norm_adj: torch.sparse_coo_tensor of shape [num_users+num_items, num_users+num_items]
    """
    num_nodes = num_items + num_users
    users = train_interactions[:, 0]
    items = train_interactions[:, 1] + num_users
    rows = torch.cat([users, items], dim=0)
    cols = torch.cat([items, users], dim=0)

    indices = torch.stack([rows, cols], dim=0).to(device=device)
    values = torch.ones(indices.shape[1], device=device)
    adj = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes), device=device)

    degrees = torch.sparse.sum(adj, dim=1).to_dense()
    degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)
    norm_values = 1.0 / (torch.sqrt(degrees[rows]) * torch.sqrt(degrees[cols])).to(device)
    norm_adj = torch.sparse_coo_tensor(indices, norm_values, size=(num_nodes, num_nodes), device=device)
    return norm_adj
