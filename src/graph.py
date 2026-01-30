import torch


def build_normalized_adj(
    num_users: int,
    num_items: int,
    train_interactions: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    train_interactions = torch.unique(train_interactions, dim=0)
    num_nodes = num_users + num_items

    users = train_interactions[:, 0].long()
    items = (train_interactions[:, 1].long() + num_users)

    # put index tensors on the target device
    users = users.to(device)
    items = items.to(device)

    rows = torch.cat([users, items], dim=0)
    cols = torch.cat([items, users], dim=0)

    indices = torch.stack([rows, cols], dim=0)  # shape (2, E)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)

    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device)
    adj = adj.coalesce()

    # degree: sum over columns (or rows) — gives shape (num_nodes,)
    degrees = torch.sparse.sum(adj, dim=1).to_dense()  # float tensor on device
    # replace zeros to avoid division by zero (isolated nodes)
    degrees = torch.where(degrees == 0.0, torch.ones_like(degrees), degrees)

    inv_sqrt_deg = 1.0 / torch.sqrt(degrees)  # shape (num_nodes,)

    # compute normalized values for each edge = inv_sqrt_deg[row] * inv_sqrt_deg[col]
    norm_values = inv_sqrt_deg[rows] * inv_sqrt_deg[cols]
    norm_adj = torch.sparse_coo_tensor(indices, norm_values, (num_nodes, num_nodes), device=device)
    norm_adj = norm_adj.coalesce()
    return norm_adj
