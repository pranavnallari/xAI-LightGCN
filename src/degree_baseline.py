# degree_baseline.py

import torch
import copy
import numpy as np

from graph import build_adj
from counterfactual import get_scores, get_item_at_rank, get_rank_of


# ------------------------------------------------
# Degree computation
# ------------------------------------------------

def compute_item_degrees(train_dict):
    """
    Computes the degree (number of users who interacted with it)
    for every item in the training set.

    Returns:
        item_degree: {item_id: degree_count}
    """
    item_degree = {}
    for user, items in train_dict.items():
        for item in items:
            item_degree[item] = item_degree.get(item, 0) + 1
    return item_degree


# ------------------------------------------------
# Degree-based attribution
# ------------------------------------------------

def degree_contributions(user_id, train_dict, item_degree, top_k=5):
    """
    Ranks a user's training items by their global degree (popularity).

    The intuition: high-degree items connect to many users in the graph,
    giving them larger influence in LightGCN's neighbourhood aggregation.
    This is the naive structural baseline — no learning required.

    Returns:
        list of (item_id, degree) sorted by degree descending, top-k only
    """
    if user_id not in train_dict:
        return []

    scored = [
        (item, item_degree.get(item, 0))
        for item in train_dict[user_id]
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ------------------------------------------------
# Full baseline evaluation (mirrors counterfactual.py structure)
# ------------------------------------------------

def run_degree_baseline(
    model,
    train_dict,
    explainer,
    num_users=10,
    top_k=5,
    test_ranks=(1, 5, 10, 20),
):
    """
    Degree-based counterfactual edge-removal baseline.

    Mirrors run_counterfactual_tests() exactly, but uses degree_contributions()
    instead of one_hop_contributions() to select which edges to remove.

    This lets us produce an apples-to-apples comparison:
    - Our method: removes top-k edges by propagation-path attribution score
    - Degree baseline: removes top-k edges by item popularity (degree)
    - Random baseline: removes top-k edges uniformly at random

    The key metric is Avg |Score Drop| — if our method outperforms degree,
    it means propagation-path analysis captures influence beyond simple
    popularity, which is the core claim of the paper.

    Returns:
        results: list of dicts, one per (user, rank_position, removed_item)
    """
    print("\n--- Degree-Based Baseline (Counterfactual) ---")
    print(f"Testing rank positions: {list(test_ranks)}")
    print(f"Users: {num_users} | Top-k edges per item: {top_k}\n")

    item_degree = compute_item_degrees(train_dict)
    users  = list(train_dict.keys())[:num_users]
    device = model.adj.device
    results = []

    for u in users:
        base_scores = get_scores(model, u, train_dict)

        # ---- Collect target items at each rank ----
        rank_targets = {}
        for rank in test_ranks:
            item_id, score = get_item_at_rank(base_scores, rank)
            if item_id is not None:
                rank_targets[rank] = (item_id, score)

        if not rank_targets:
            continue

        print(f"{'='*60}")
        print(f"User {u}")
        for rank, (item_id, score) in rank_targets.items():
            print(f"  Rank {rank:2d}: {explainer.fmt(item_id)}  "
                  f"[score={score:.4f}]")
        print()

        for rank, (target_item, base_score) in rank_targets.items():

            deg_contributors = degree_contributions(
                u, train_dict, item_degree, top_k
            )
            if not deg_contributors:
                continue

            print(f"  -- Rank {rank} | Degree-based removals --")

            for item_j, degree in deg_contributors:

                # ---- Build counterfactual graph ----
                new_train = {k: set(v) for k, v in train_dict.items()}
                new_train[u].discard(item_j)

                new_train_data = [
                    (uu, ii)
                    for uu, items in new_train.items()
                    for ii in items
                ]

                new_adj = build_adj(
                    model.num_users, model.num_items,
                    new_train_data, device=device,
                )

                # ---- Adj swap ----
                original_adj = model.adj
                model.adj    = new_adj

                cf_scores   = get_scores(model, u, new_train)
                cf_score    = cf_scores[target_item].item()
                cf_rank     = get_rank_of(cf_scores, target_item)

                model.adj = original_adj

                # ---- Record ----
                score_drop  = base_score - cf_score
                rank_change = cf_rank - rank

                rank_str = (
                    f"{rank}->{cf_rank} "
                    f"({'down' if rank_change > 0 else 'up' if rank_change < 0 else '='}"
                    f" {abs(rank_change)})"
                )

                print(
                    f"    Remove {explainer.fmt(item_j)}  "
                    f"[degree={degree}]\n"
                    f"      Score drop: {score_drop:+.4f} | "
                    f"Rank: {rank_str}"
                )

                results.append({
                    "method":       "degree",
                    "user":         u,
                    "target_item":  target_item,
                    "target_rank":  rank,
                    "removed_item": item_j,
                    "item_degree":  degree,
                    "base_score":   base_score,
                    "cf_score":     cf_score,
                    "score_drop":   score_drop,
                    "base_rank":    rank,
                    "cf_rank":      cf_rank,
                    "rank_change":  rank_change,
                })

            print()

    _print_degree_summary(results, test_ranks)
    return results


# ------------------------------------------------
# Summary
# ------------------------------------------------

def _print_degree_summary(results, test_ranks):
    if not results:
        return

    print(f"\n{'='*60}")
    print("DEGREE BASELINE SUMMARY")
    print(f"{'Rank':>6} | {'N':>5} | {'Avg |Score Drop|':>16} | "
          f"{'Rank Changes':>12}")
    print("-" * 50)

    for rank in sorted(test_ranks):
        subset = [r for r in results if r["target_rank"] == rank]
        if not subset:
            continue
        n            = len(subset)
        avg_abs_drop = np.mean([abs(r["score_drop"]) for r in subset])
        rank_changed = sum(1 for r in subset if r["rank_change"] != 0)
        print(f"{rank:>6} | {n:>5} | {avg_abs_drop:>16.4f} | "
              f"{rank_changed:>5}/{n} changed")

    n_all        = len(results)
    avg_abs_all  = np.mean([abs(r["score_drop"]) for r in results])
    rc_all       = sum(1 for r in results if r["rank_change"] != 0)
    print(f"{'ALL':>6} | {n_all:>5} | {avg_abs_all:>16.4f} | "
          f"{rc_all:>5}/{n_all} changed")
