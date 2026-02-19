# counterfactual.py

import torch
import copy
import numpy as np

from graph import build_adj


def get_top_item(model, user_id, train_dict):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model()
        scores = torch.matmul(item_emb, user_emb[user_id])
        scores[list(train_dict[user_id])] = -1e9

        top_item = torch.topk(scores, k=1).indices.item()
        top_score = scores[top_item].item()
        rank = (scores > scores[top_item]).sum().item() + 1

    return top_item, top_score, rank


def run_counterfactual_tests(
    model,
    train_dict,
    explainer,
    num_users=100,
    top_k=5,
):
    """
    Performs counterfactual edge-removal sensitivity analysis.

    For each user, removes one training item at a time from the graph,
    rebuilds the adjacency matrix, and measures the change in the top
    recommended item's score and rank.

    NOTE ON APPROXIMATION: This is a structural sensitivity test — the model
    weights (embeddings) are NOT retrained after each edge removal. Only the
    propagation graph is changed. This approximates the effect of the edge on
    the final embeddings via message passing, but does not reflect how the
    model would learn differently had the edge never existed. For a true
    counterfactual, the model would need to be retrained from scratch for each
    removal — which is computationally prohibitive. This approximation is
    standard in GNN explainability literature (cf. GNNExplainer).
    """
    print("\n--- Counterfactual Tests ---")

    users = list(train_dict.keys())[:num_users]
    results = []

    # Extract the device once from the existing adjacency tensor
    device = model.adj.device

    for u in users:

        rec_item, base_score, base_rank = get_top_item(model, u, train_dict)

        contributors = explainer.one_hop_contributions(u, rec_item, top_k)
        if not contributors:
            continue

        print(f"\nUser {u}")
        print(f"Recommended: {explainer.fmt(rec_item)}")
        print(f"Base score: {base_score:.4f} | Rank: {base_rank}")

        for item_j, _ in contributors:

            # ---- Remove edge ----
            new_train = copy.deepcopy(train_dict)
            if item_j in new_train[u]:
                new_train[u].remove(item_j)

            # ---- Convert to edge list ----
            new_train_data = [
                (uu, ii)
                for uu, items in new_train.items()
                for ii in items
            ]

            new_adj = build_adj(
                model.num_users,
                model.num_items,
                new_train_data,
                device=device,
            )

            original_adj = model.adj
            model.adj = new_adj

            # ---- Recompute recommendation ----
            rec2, score2, rank2 = get_top_item(model, u, new_train)

            # ---- Restore original adj ----
            model.adj = original_adj

            drop = base_score - score2

            print(
                f"Remove {explainer.fmt(item_j)} "
                f"| Score drop {drop:.4f} "
                f"| Rank {base_rank}->{rank2}"
            )

            results.append((u, rec_item, item_j, drop, base_rank, rank2))

    return results
