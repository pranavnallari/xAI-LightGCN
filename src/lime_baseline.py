# lime_baseline.py

import torch
import numpy as np
import random
from sklearn.linear_model import Ridge

from graph import build_adj
from counterfactual import get_scores, get_item_at_rank, get_rank_of


# ------------------------------------------------
# LIME perturbation engine
# ------------------------------------------------

def _build_perturbed_adj(user_id, full_item_list, mask, train_dict,
                          num_users, num_items, device):
    """
    Builds an adjacency matrix with a subset of the user's items masked out.
    mask: binary array of length len(full_item_list), 1 = keep, 0 = remove
    """
    new_train = {k: set(v) for k, v in train_dict.items()}
    for item, keep in zip(full_item_list, mask):
        if not keep:
            new_train[user_id].discard(item)

    new_train_data = [
        (u, i) for u, items in new_train.items() for i in items
    ]

    return build_adj(num_users, num_items, new_train_data, device=device), new_train


def lime_contributions(
    model,
    user_id,
    target_item,
    train_dict,
    top_k=5,
    n_perturbations=100,
    mask_prob=0.5,
    random_seed=42,
):
    """
    LIME attribution for a single (user, target_item) pair.

    Procedure:
      1. Sample n_perturbations binary masks over the user's training items
      2. For each mask, rebuild the graph and score the target item
      3. Fit a weighted Ridge regression: mask -> score
      4. Regression coefficients = item importance scores
      5. Return top-k items by |coefficient|

    Weighting: perturbations closer to the original (fewer removals) are
    weighted more heavily using an exponential kernel, standard in LIME.

    Returns:
        list of (item_id, coefficient) sorted by |coefficient| descending
    """
    rng = np.random.RandomState(random_seed)
    device = model.adj.device

    user_items = list(train_dict.get(user_id, set()))
    n_items = len(user_items)

    if n_items == 0:
        return []

    # ---- Sample masks ----
    # Each row is a binary vector: 1=keep item, 0=remove item
    masks = rng.binomial(1, 1 - mask_prob, size=(n_perturbations, n_items))
    # Always include the original (all ones) as first sample
    masks[0] = np.ones(n_items)

    scores = []
    original_adj = model.adj

    model.eval()
    with torch.no_grad():
        for mask in masks:
            new_adj, new_train = _build_perturbed_adj(
                user_id, user_items, mask, train_dict,
                model.num_users, model.num_items, device
            )
            model.adj = new_adj
            sc = get_scores(model, user_id, new_train)
            scores.append(sc[target_item].item())

    model.adj = original_adj

    scores = np.array(scores)

    # ---- Exponential kernel weights ----
    # Similarity = fraction of items kept (cosine to all-ones vector)
    similarities = masks.sum(axis=1) / n_items
    kernel_width = 0.25
    weights = np.exp(-(1 - similarities) ** 2 / (2 * kernel_width ** 2))

    # ---- Weighted Ridge regression ----
    reg = Ridge(alpha=1.0, fit_intercept=True)
    reg.fit(masks, scores, sample_weight=weights)
    coefficients = reg.coef_  # shape: (n_items,)

    # ---- Rank by |coefficient| ----
    item_scores = list(zip(user_items, coefficients))
    item_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    return item_scores[:top_k]


# ------------------------------------------------
# Full LIME baseline evaluation
# ------------------------------------------------

def run_lime_baseline(
    model,
    train_dict,
    explainer,
    num_users=10,
    top_k=5,
    test_ranks=(1, 5, 10, 20),
    n_perturbations=100,
    mask_prob=0.5,
    random_seed=42,
):
    """
    LIME-based counterfactual edge-removal baseline.

    Mirrors run_counterfactual_tests() and run_degree_baseline() exactly,
    replacing attribution with LIME coefficients.

    LIME treats LightGCN as a black box — it has no knowledge of graph
    structure, propagation layers, or neighbourhood aggregation. It learns
    item importance purely from input-output perturbation behaviour.

    This makes it a strong model-agnostic baseline: if our propagation-path
    method outperforms LIME, it demonstrates that graph-structural knowledge
    adds value beyond black-box probing.

    Returns:
        results: list of dicts, one per (user, rank_position, removed_item)
    """
    print("\n--- LIME Baseline (Counterfactual) ---")
    print(f"Testing rank positions: {list(test_ranks)}")
    print(f"Users: {num_users} | Top-k: {top_k} | "
          f"Perturbations per target: {n_perturbations}\n")

    users  = list(train_dict.keys())[:num_users]
    device = model.adj.device
    results = []

    for u_idx, u in enumerate(users, 1):
        base_scores = get_scores(model, u, train_dict)

        # ---- Collect target items at each rank ----
        rank_targets = {}
        for rank in test_ranks:
            item_id, score = get_item_at_rank(base_scores, rank)
            if item_id is not None:
                rank_targets[rank] = (item_id, score)

        if not rank_targets:
            continue

        print(f"[{u_idx:2d}/{num_users}] User {u}")
        for rank, (item_id, score) in rank_targets.items():
            print(f"  Rank {rank:2d}: {explainer.fmt(item_id)}  "
                  f"[score={score:.4f}]")
        print()

        for rank, (target_item, base_score) in rank_targets.items():

            # ---- LIME attribution ----
            lime_contribs = lime_contributions(
                model, u, target_item, train_dict,
                top_k=top_k,
                n_perturbations=n_perturbations,
                mask_prob=mask_prob,
                random_seed=random_seed + u_idx + rank,
            )

            if not lime_contribs:
                continue

            print(f"  -- Rank {rank} | LIME removals --")

            for item_j, coef in lime_contribs:

                # ---- Counterfactual edge removal ----
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

                original_adj = model.adj
                model.adj    = new_adj

                cf_scores   = get_scores(model, u, new_train)
                cf_score    = cf_scores[target_item].item()
                cf_rank     = get_rank_of(cf_scores, target_item)

                model.adj = original_adj

                score_drop  = base_score - cf_score
                rank_change = cf_rank - rank

                rank_str = (
                    f"{rank}->{cf_rank} "
                    f"({'down' if rank_change > 0 else 'up' if rank_change < 0 else '='}"
                    f" {abs(rank_change)})"
                )

                print(
                    f"    Remove {explainer.fmt(item_j)}  "
                    f"[LIME coef={coef:+.4f}]\n"
                    f"      Score drop: {score_drop:+.4f} | Rank: {rank_str}"
                )

                results.append({
                    "method":       "lime",
                    "user":         u,
                    "target_item":  target_item,
                    "target_rank":  rank,
                    "removed_item": item_j,
                    "lime_coef":    coef,
                    "base_score":   base_score,
                    "cf_score":     cf_score,
                    "score_drop":   score_drop,
                    "base_rank":    rank,
                    "cf_rank":      cf_rank,
                    "rank_change":  rank_change,
                })

            print()

    _print_lime_summary(results, test_ranks)
    return results


# ------------------------------------------------
# Summary + full 3-way comparison
# ------------------------------------------------

def _print_lime_summary(results, test_ranks):
    if not results:
        return

    print(f"\n{'='*60}")
    print("LIME BASELINE SUMMARY")
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

    n_all       = len(results)
    avg_abs_all = np.mean([abs(r["score_drop"]) for r in results])
    rc_all      = sum(1 for r in results if r["rank_change"] != 0)
    print(f"{'ALL':>6} | {n_all:>5} | {avg_abs_all:>16.4f} | "
          f"{rc_all:>5}/{n_all} changed")


def print_three_way_comparison(attributed, lime_res, degree_res, random_res,
                                test_ranks):
    """
    Prints the full 3-way comparison table across all methods.
    Call this from main.py after all four methods have been run.
    """
    print(f"\n{'='*70}")
    print("FULL METHOD COMPARISON: Avg |Score Drop| by Rank Position")
    print(f"{'Rank':>6} | {'Ours':>8} | {'LIME':>8} | "
          f"{'Degree':>8} | {'Random':>8} | {'Lift(Ours/LIME)':>15}")
    print("-" * 70)

    for rank in sorted(test_ranks):
        def avg(res):
            s = [r for r in res if r["target_rank"] == rank]
            return np.mean([abs(r["score_drop"]) for r in s]) if s else 0.0

        a = avg(attributed)
        l = avg(lime_res)
        d = avg(degree_res)
        r = avg(random_res)
        lift = a / l if l > 0 else float("inf")

        print(f"{rank:>6} | {a:>8.4f} | {l:>8.4f} | "
              f"{d:>8.4f} | {r:>8.4f} | {lift:>15.2f}x")

    # Overall
    def avg_all(res):
        return np.mean([abs(r["score_drop"]) for r in res]) if res else 0.0

    a = avg_all(attributed)
    l = avg_all(lime_res)
    d = avg_all(degree_res)
    r = avg_all(random_res)
    lift = a / l if l > 0 else float("inf")

    print("-" * 70)
    print(f"{'ALL':>6} | {a:>8.4f} | {l:>8.4f} | "
          f"{d:>8.4f} | {r:>8.4f} | {lift:>15.2f}x")

    print(f"\nOurs vs LIME lift:   {a/l:.2f}x")
    print(f"Ours vs Degree lift: {a/d:.2f}x")
    print(f"Ours vs Random lift: {a/r:.2f}x")
