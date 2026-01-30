# eval.py
# Correct evaluation for LightGCN (Temporal / Leave-One-Out compatible)

import torch
import numpy as np
from collections import defaultdict


def evaluate_lightgcn(model, train_interactions, test_interactions, num_items, k=10):
    model.eval()
    device = next(model.parameters()).device

    if test_interactions.numel() == 0:
        print("⚠️ Empty test set!")
        return {f'{m}@10': 0.0 for m in ['Hit', 'Recall', 'NDCG', 'Precision']}

    print(f"🔍 Eval: {test_interactions.size(0)} test edges")

    # ---- user → train items
    user_train_items = defaultdict(set)
    for u, i in train_interactions.cpu().tolist():
        user_train_items[int(u)].add(int(i))

    # ---- user → test items
    user_test_items = defaultdict(list)
    for u, i in test_interactions.cpu().tolist():
        user_test_items[int(u)].append(int(i))

    metrics = {
        'Hit@10': [],
        'Recall@10': [],
        'NDCG@10': [],
        'Precision@10': []
    }

    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings()
        scores = torch.matmul(user_emb, item_emb.T)  # [num_users, num_items]

        for u, test_items in user_test_items.items():
            test_set = set(test_items)
            if not test_set:
                continue

            # 🔑 CLONE before masking
            scores_u = scores[u].clone()

            train_items = torch.tensor(
                list(user_train_items[u]),
                device=device,
                dtype=torch.long
            )
            scores_u[train_items] = -1e9

            topk_items = torch.topk(scores_u, k).indices.tolist()

            # ---- Hit / Recall / Precision
            hits = sum(1 for item in topk_items if item in test_set)
            metrics['Hit@10'].append(1.0 if hits > 0 else 0.0)
            metrics['Recall@10'].append(hits / len(test_set))
            metrics['Precision@10'].append(hits / k)

            # ---- NDCG@K (correct)
            dcg = 0.0
            for rank, item in enumerate(topk_items):
                if item in test_set:
                    dcg += 1.0 / np.log2(rank + 2)

            idcg = sum(
                1.0 / np.log2(r + 2)
                for r in range(min(len(test_set), k))
            )
            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics['NDCG@10'].append(ndcg)

    return {m: float(np.mean(v)) for m, v in metrics.items()}
