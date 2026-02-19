# counterfactual.py

import torch
import copy
import numpy as np
import random

from graph import build_adj


# ------------------------------------------------
# Core scoring utilities
# ------------------------------------------------

def get_scores(model, user_id, train_dict):
    """
    Returns the full score vector for a user with training items masked.
    """
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model()
        scores = torch.matmul(item_emb, user_emb[user_id])
        scores[list(train_dict[user_id])] = -1e9
    return scores


def get_item_at_rank(scores, target_rank):
    """
    Returns (item_id, score) for the item sitting at a given rank position.
    Rank is 1-indexed. Returns None if rank exceeds number of valid items.
    """
    k = min(target_rank, (scores > -1e8).sum().item())
    if k < target_rank:
        return None, None
    top = torch.topk(scores, k=k)
    item_id = top.indices[-1].item()
    score = top.values[-1].item()
    return item_id, score


def get_rank_of(scores, item_id):
    """
    Returns the current rank (1-indexed) of a specific item in the score vector.
    """
    return (scores > scores[item_id]).sum().item() + 1


# ------------------------------------------------
# Single edge removal (shared by attributed + random)
# ------------------------------------------------

def _remove_edge_and_score(model, user_id, target_item, item_to_remove,
                            train_dict, device):
    """
    Removes one (user, item_to_remove) edge, rebuilds adj, returns
    (cf_score, cf_rank) for target_item. Restores original adj afterwards.
    """
    new_train = copy.deepcopy(train_dict)
    if item_to_remove in new_train[user_id]:
        new_train[user_id].remove(item_to_remove)

    new_train_data = [
        (uu, ii)
        for uu, items in new_train.items()
        for ii in items
    ]

    new_adj = build_adj(
        model.num_users, model.num_items, new_train_data, device=device
    )

    original_adj = model.adj
    model.adj = new_adj

    cf_scores = get_scores(model, user_id, new_train)
    cf_score  = cf_scores[target_item].item()
    cf_rank   = get_rank_of(cf_scores, target_item)

    model.adj = original_adj

    return cf_score, cf_rank


# ------------------------------------------------
# Multi-rank counterfactual with random baseline
# ------------------------------------------------

def run_counterfactual_tests(
    model,
    train_dict,
    explainer,
    num_users=10,
    top_k=5,
    test_ranks=(1, 5, 10, 20),
    random_seed=42,
):
    """
    Multi-rank counterfactual edge-removal with random baseline comparison.

    For each (user, rank position) pair we run two removal strategies:

      ATTRIBUTED  — remove the top-k items identified by one_hop_contributions()
      RANDOM      — remove top_k items sampled uniformly from the user's
                    training history (excluding the attributed set, so the
                    comparison is fair)

    The summary table reports average |score drop| and rank-change rate for
    both strategies side-by-side. A meaningful attribution method should
    produce consistently larger score drops than random removal.

    WHY |score drop| NOT score drop:
    Some removals increase scores (suppressive edges). Using the absolute value
    captures edge influence regardless of direction, which is the right metric
    for comparing how "impactful" a removal strategy is.

    Returns:
        attributed_results: list of dicts (method="attributed")
        random_results:     list of dicts (method="random")
    """
    print("\n--- Multi-Rank Counterfactual Tests (with Random Baseline) ---")
    print(f"Testing rank positions: {list(test_ranks)}")
    print(f"Users: {num_users} | k per strategy: {top_k}\n")

    rng = random.Random(random_seed)
    users = list(train_dict.keys())[:num_users]
    device = model.adj.device

    attributed_results = []
    random_results     = []

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
            print(f"  Rank {rank:2d}: {explainer.fmt(item_id)}  [score={score:.4f}]")
        print()

        for rank, (target_item, base_score) in rank_targets.items():

            # -------- ATTRIBUTED removals --------
            contributors = explainer.one_hop_contributions(u, target_item, top_k)
            if not contributors:
                continue

            attributed_items = [item_j for item_j, _ in contributors]

            print(f"  -- Rank {rank} | Attributed removals --")
            for item_j, attr_score in contributors:
                cf_score, cf_rank = _remove_edge_and_score(
                    model, u, target_item, item_j, train_dict, device
                )
                score_drop = base_score - cf_score
                rank_change = cf_rank - rank

                _print_removal(explainer, item_j, score_drop, rank, cf_rank,
                               rank_change, attr_score, label="  ")

                attributed_results.append(_make_result(
                    u, target_item, rank, item_j,
                    base_score, cf_score, score_drop, cf_rank, rank_change,
                    attr_score, method="attributed"
                ))

            # -------- RANDOM removals --------
            # Sample from training items excluding the attributed set
            pool = list(train_dict[u] - set(attributed_items))
            if not pool:
                # fall back to full history if pool is empty
                pool = list(train_dict[u])

            k_rand = min(top_k, len(pool))
            random_items = rng.sample(pool, k_rand)

            print(f"\n  -- Rank {rank} | Random removals --")
            for item_j in random_items:
                cf_score, cf_rank = _remove_edge_and_score(
                    model, u, target_item, item_j, train_dict, device
                )
                score_drop = base_score - cf_score
                rank_change = cf_rank - rank

                _print_removal(explainer, item_j, score_drop, rank, cf_rank,
                               rank_change, attr_score=None, label="  ")

                random_results.append(_make_result(
                    u, target_item, rank, item_j,
                    base_score, cf_score, score_drop, cf_rank, rank_change,
                    attr_score=None, method="random"
                ))

            print()

    # ---- Comparison summary ----
    _print_comparison_summary(attributed_results, random_results, test_ranks)

    return attributed_results, random_results


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def _make_result(user, target_item, rank, removed_item,
                 base_score, cf_score, score_drop, cf_rank, rank_change,
                 attr_score, method):
    return {
        "method":          method,
        "user":            user,
        "target_item":     target_item,
        "target_rank":     rank,
        "removed_item":    removed_item,
        "base_score":      base_score,
        "cf_score":        cf_score,
        "score_drop":      score_drop,
        "base_rank":       rank,
        "cf_rank":         cf_rank,
        "rank_change":     rank_change,
        "attribution_score": attr_score,
    }


def _print_removal(explainer, item_j, score_drop, base_rank, cf_rank,
                   rank_change, attr_score, label=""):
    rank_str = (
        f"{base_rank}->{cf_rank} "
        f"({'down' if rank_change > 0 else 'up' if rank_change < 0 else '='}"
        f" {abs(rank_change)})"
    )
    attr_str = f" | Attr: {attr_score:.4f}" if attr_score is not None else ""
    print(
        f"{label}  Remove {explainer.fmt(item_j)}\n"
        f"{label}    Score drop: {score_drop:+.4f} | Rank: {rank_str}{attr_str}"
    )


def _print_comparison_summary(attributed, random_res, test_ranks):
    """
    Side-by-side summary: attributed vs random, broken down by rank position.
    The key metric is Avg |Score Drop| — higher means more impactful removals.
    Also reports lift: how many times more impactful attributed is vs random.
    """
    if not attributed or not random_res:
        return

    print(f"\n{'='*60}")
    print("BASELINE COMPARISON: Attributed vs Random Edge Removal")
    print(f"Metric: Avg |Score Drop| (higher = more impactful removal)\n")

    header = (f"{'Rank':>6} | {'Attr |Drop|':>11} | {'Rand |Drop|':>11} | "
              f"{'Lift':>6} | {'Attr RC%':>9} | {'Rand RC%':>9}")
    print(header)
    print("-" * len(header))

    all_lifts = []

    for rank in sorted(test_ranks):
        a = [r for r in attributed  if r["target_rank"] == rank]
        b = [r for r in random_res  if r["target_rank"] == rank]
        if not a or not b:
            continue

        a_abs  = np.mean([abs(r["score_drop"]) for r in a])
        b_abs  = np.mean([abs(r["score_drop"]) for r in b])
        lift   = a_abs / b_abs if b_abs > 0 else float("inf")
        a_rc   = 100 * sum(1 for r in a if r["rank_change"] != 0) / len(a)
        b_rc   = 100 * sum(1 for r in b if r["rank_change"] != 0) / len(b)

        all_lifts.append(lift)
        print(f"{rank:>6} | {a_abs:>11.4f} | {b_abs:>11.4f} | "
              f"{lift:>6.2f}x | {a_rc:>8.1f}% | {b_rc:>8.1f}%")

    # Overall row
    a_abs_all = np.mean([abs(r["score_drop"]) for r in attributed])
    b_abs_all = np.mean([abs(r["score_drop"]) for r in random_res])
    lift_all  = a_abs_all / b_abs_all if b_abs_all > 0 else float("inf")
    a_rc_all  = 100 * sum(1 for r in attributed  if r["rank_change"] != 0) / len(attributed)
    b_rc_all  = 100 * sum(1 for r in random_res  if r["rank_change"] != 0) / len(random_res)

    print("-" * len(header))
    print(f"{'ALL':>6} | {a_abs_all:>11.4f} | {b_abs_all:>11.4f} | "
          f"{lift_all:>6.2f}x | {a_rc_all:>8.1f}% | {b_rc_all:>8.1f}%")

    print(f"\nOverall lift: attributed removal is {lift_all:.2f}x more impactful than random.")
    print(f"Rank-change rate: {a_rc_all:.1f}% (attributed) vs {b_rc_all:.1f}% (random).")
