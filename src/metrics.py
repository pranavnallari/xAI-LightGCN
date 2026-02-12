# metrics.py

import torch
import numpy as np
from collections import defaultdict


def evaluate_lightgcn(
    model,
    train_interactions,
    test_interactions,
    num_items,
    k=10,
    use_full_matrix=True,   # Set False for per‑user scoring (recommended for large datasets)
    verbose=True
):
    """Evaluate model using full ranking protocol.

    Args:
        model: Trained LightGCN model.
        train_interactions: [num_train_edges, 2] tensor of training interactions.
        test_interactions: [num_test_edges, 2] tensor of test interactions.
        num_items: Number of items.
        k: Cutoff for top‑k evaluation.
        use_full_matrix: If True, compute full [num_users, num_items] score matrix.
                         If False, compute per‑user scores (memory efficient).
        verbose: If True, print evaluation info.

    Returns:
        Dictionary of metric averages (Hit, Recall, NDCG, Precision) @ k.
    """
    model.eval()
    device = next(model.parameters()).device

    if test_interactions.numel() == 0:
        if verbose:
            print("⚠️ Empty test set!")
        return {f'{m}@{k}': 0.0 for m in ['Hit', 'Recall', 'NDCG', 'Precision']}

    if verbose:
        print(f"🔍 Eval: {test_interactions.size(0)} test edges")

    # ---- User → training items (set for O(1) membership) ----
    user_train_items = defaultdict(set)
    for u, i in train_interactions.cpu().tolist():
        user_train_items[int(u)].add(int(i))

    # ---- User → test items (list to preserve duplicates) ----
    user_test_items = defaultdict(list)
    for u, i in test_interactions.cpu().tolist():
        user_test_items[int(u)].append(int(i))

    metrics = {
        f'Hit@{k}': [],
        f'Recall@{k}': [],
        f'NDCG@{k}': [],
        f'Precision@{k}': []
    }

    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings()

        # ---- Score computation ----
        if use_full_matrix:
            # Full score matrix – simple but memory intensive
            if verbose and user_emb.size(0) * item_emb.size(0) > 10_000_000:
                print("⚠️ Large score matrix. Consider setting `use_full_matrix=False`.")
            scores = torch.matmul(user_emb, item_emb.T)  # [num_users, num_items]
        else:
            scores = None  # Will compute per user

        # ---- Per‑user evaluation ----
        for u, test_item_list in user_test_items.items():
            if not test_item_list:
                continue

            # Get scores for this user
            if use_full_matrix:
                scores_u = scores[u].clone()
            else:
                scores_u = torch.matmul(user_emb[u], item_emb.T)  # [num_items]

            # ---- Mask training items ----
            if u in user_train_items:
                train_items = torch.tensor(
                    list(user_train_items[u]),
                    device=device,
                    dtype=torch.long
                )
                scores_u[train_items] = -float('inf')

            # ---- Determine effective k ----
            k_effective = min(k, scores_u.numel())
            if k_effective == 0:
                continue

            # ---- Get top‑k items ----
            topk_items = torch.topk(scores_u, k_effective).indices.cpu().tolist()

            # ---- Test set membership ----
            test_set = set(test_item_list)          # for O(1) membership
            test_list = test_item_list              # for counting duplicates

            # ---- Hit (binary) ----
            hits_binary = 1.0 if any(item in test_set for item in topk_items) else 0.0
            metrics[f'Hit@{k}'].append(hits_binary)

            # ---- Recall (fraction of all test items found) ----
            hits_count = sum(1 for item in topk_items if item in test_set)
            metrics[f'Recall@{k}'].append(hits_count / len(test_list))

            # ---- Precision (hits / k_effective) ----
            metrics[f'Precision@{k}'].append(hits_count / k_effective)

            # ---- NDCG (with duplicate relevance) ----
            dcg = 0.0
            for rank, item in enumerate(topk_items):
                if item in test_set:
                    relevance = test_list.count(item)  # duplicates count
                    dcg += relevance / np.log2(rank + 2)

            # Ideal DCG – all test items have relevance 1
            ideal_relevances = [1] * len(test_list)
            ideal_relevances = sorted(ideal_relevances, reverse=True)[:k_effective]
            idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f'NDCG@{k}'].append(ndcg)

    # ---- Average metrics ----
    return {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}