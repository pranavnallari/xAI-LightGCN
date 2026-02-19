# graph.py

import numpy as np
import torch
import scipy.sparse as sp


def build_adj(num_users, num_items, train_data, device):
    N = num_users + num_items
    rows = []
    cols = []
    for u, i in train_data:
        rows.append(u)
        cols.append(num_users + i)
        rows.append(num_users + i)
        cols.append(u)
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
    degree = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.zeros_like(degree)
    nonzero = degree > 0
    d_inv_sqrt[nonzero] = np.power(degree[nonzero], -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
    norm_adj = norm_adj.tocoo()
    indices = torch.LongTensor(
        np.vstack((norm_adj.row, norm_adj.col))
    ).to(device)
    values = torch.FloatTensor(norm_adj.data).to(device)
    return torch.sparse_coo_tensor(
        indices,
        values,
        torch.Size([N, N])
    ).coalesce()


def build_adj_from_dict(num_users, num_items, train_dict, device):
    """
    Convenience wrapper: builds adjacency from a {user_id: set(item_ids)} dict.
    """
    interactions = [
        (u, i)
        for u, items in train_dict.items()
        for i in items
    ]
    return build_adj(num_users, num_items, interactions, device)
