# faithfulness.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from scipy.stats import pearsonr, spearmanr

from graph import build_adj
from counterfactual import get_scores


# ------------------------------------------------
# BPR fine-tuning on modified graph
# ------------------------------------------------

def _finetune(model, train_dict, train_data, device,
              epochs=5, lr=0.001, batch_size=2048):
    """
    Fine-tunes the model for a small number of epochs on the provided
    training data using BPR loss.

    Used to approximate 'actual' score drop after an edge is removed:
    we remove the edge, fine-tune briefly, then re-measure the score.
    This is faster than full retraining while still capturing how the
    model *adapts* to the structural change.

    NOTE: Modifies model in-place. Callers should pass a copy.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data = list(train_data)

    model.train()
    for _ in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            if not batch:
                continue

            users_b, pos_items_b = zip(*batch)
            users_t     = torch.LongTensor(users_b).to(device)
            pos_items_t = torch.LongTensor(pos_items_b).to(device)

            neg_items = []
            for u in users_b:
                while True:
                    neg = random.randrange(model.num_items)
                    if neg not in train_dict[u]:
                        neg_items.append(neg)
                        break
            neg_items_t = torch.LongTensor(neg_items).to(device)

            optimizer.zero_grad()
            user_emb, item_emb = model()

            u_emb   = user_emb[users_t]
            pos_emb = item_emb[pos_items_t]
            neg_emb = item_emb[neg_items_t]

            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
            loss.backward()
            optimizer.step()

    model.eval()


# ------------------------------------------------
# Main faithfulness evaluation
# ------------------------------------------------

def run_faithfulness_evaluation(
    model,
    train_dict,
    train_data,
    explainer,
    num_users=50,
    top_k=3,
    finetune_epochs=5,
    finetune_lr=0.001,
    random_seed=42,
):
    """
    Faithfulness Evaluation: Predicted vs Actual Score Drop.

    For each sampled user:
      1. Get their top-1 recommended item.
      2. Get top-k attributed edges via one_hop_contributions().
      3. For each edge:
         a. PREDICTED drop  — structural sensitivity (adj-swap, no retraining).
         b. ACTUAL drop     — deep-copy model, remove edge, fine-tune for
                              `finetune_epochs` epochs, re-measure score.
      4. Collect all (predicted, actual) pairs and compute:
         - Pearson r  (linear correlation)
         - Spearman r (rank correlation — more robust to outliers)

    WHY THIS MATTERS:
    The structural sensitivity approximation is fast but assumes embedding
    weights don't need to change after an edge removal. Faithfulness measures
    how well this approximation tracks what actually happens when the model
    adapts. A high correlation (r > 0.5) means the approximation is reliable
    enough to trust for explanation purposes.

    Returns:
        results:  list of dicts with predicted/actual drops per edge
        pearson_r, spearman_r: overall correlation coefficients
    """
    print("\n--- Faithfulness Evaluation ---")
    print(f"Users: {num_users} | Attributed edges per user: {top_k}")
    print(f"Fine-tune epochs per removal: {finetune_epochs}\n")

    rng = random.Random(random_seed)
    device = model.adj.device

    # Sample users that have enough training items
    eligible = [u for u, items in train_dict.items() if len(items) >= top_k + 2]
    rng.shuffle(eligible)
    users = eligible[:num_users]

    results = []

    for idx, u in enumerate(users, 1):

        # ---- Base score for top-1 item ----
        base_scores = get_scores(model, u, train_dict)
        target_item = torch.topk(base_scores, k=1).indices.item()
        base_score  = base_scores[target_item].item()

        contributors = explainer.one_hop_contributions(u, target_item, top_k)
        if not contributors:
            continue

        print(f"[{idx:3d}/{num_users}] User {u} | "
              f"Target: {explainer.fmt(target_item)}")

        for item_j, attr_score in contributors:

            # ---- PREDICTED drop (structural sensitivity, fast) ----
            new_train = {k: set(v) for k, v in train_dict.items()}
            new_train[u].discard(item_j)

            new_train_data = [
                (uu, ii) for uu, items in new_train.items() for ii in items
            ]

            new_adj = build_adj(
                model.num_users, model.num_items, new_train_data, device=device
            )

            original_adj = model.adj
            model.adj = new_adj
            cf_scores_struct = get_scores(model, u, new_train)
            predicted_drop = base_score - cf_scores_struct[target_item].item()
            model.adj = original_adj

            # ---- ACTUAL drop (fine-tuned, slow) ----
            model_ft = copy.deepcopy(model)
            model_ft.adj = new_adj

            _finetune(
                model_ft, new_train, new_train_data, device,
                epochs=finetune_epochs, lr=finetune_lr,
            )

            with torch.no_grad():
                ft_scores = get_scores(model_ft, u, new_train)
                actual_drop = base_score - ft_scores[target_item].item()

            del model_ft

            print(
                f"  Remove {explainer.fmt(item_j)}\n"
                f"    Predicted drop: {predicted_drop:+.4f} | "
                f"Actual drop: {actual_drop:+.4f} | "
                f"Attr score: {attr_score:.4f}"
            )

            results.append({
                "user":           u,
                "target_item":    target_item,
                "removed_item":   item_j,
                "attr_score":     attr_score,
                "predicted_drop": predicted_drop,
                "actual_drop":    actual_drop,
            })

    # ---- Correlation ----
    if len(results) < 3:
        print("Not enough results to compute correlation.")
        return results, None, None

    predicted = np.array([r["predicted_drop"] for r in results])
    actual    = np.array([r["actual_drop"]    for r in results])

    pearson_r,  pearson_p  = pearsonr(predicted, actual)
    spearman_r, spearman_p = spearmanr(predicted, actual)

    _print_faithfulness_summary(results, pearson_r, pearson_p,
                                 spearman_r, spearman_p)

    return results, pearson_r, spearman_r


# ------------------------------------------------
# Summary
# ------------------------------------------------

def _print_faithfulness_summary(results, pearson_r, pearson_p,
                                  spearman_r, spearman_p):

    predicted = np.array([r["predicted_drop"] for r in results])
    actual    = np.array([r["actual_drop"]    for r in results])

    print(f"\n{'='*60}")
    print("FAITHFULNESS SUMMARY")
    print(f"{'='*60}")
    print(f"Total edge removals evaluated : {len(results)}")
    print(f"Avg predicted |drop|          : {np.mean(np.abs(predicted)):.4f}")
    print(f"Avg actual    |drop|          : {np.mean(np.abs(actual)):.4f}")
    print()
    print(f"Pearson  r  = {pearson_r:+.4f}  (p = {pearson_p:.4f})")
    print(f"Spearman r  = {spearman_r:+.4f}  (p = {spearman_p:.4f})")
    print()

    # Interpret
    if abs(pearson_r) >= 0.7:
        strength = "strong"
    elif abs(pearson_r) >= 0.5:
        strength = "moderate"
    elif abs(pearson_r) >= 0.3:
        strength = "weak but meaningful"
    else:
        strength = "weak"

    print(f"Interpretation: {strength} correlation between structural "
          f"sensitivity predictions and post-fine-tuning actual score drops.")
    print(
        f"This suggests the structural approximation is "
        f"{'reliable' if abs(pearson_r) >= 0.5 else 'a rough guide'} "
        f"for identifying influential edges without retraining."
    )

    # Sign agreement
    same_sign = np.sum(np.sign(predicted) == np.sign(actual))
    print(f"\nSign agreement: {same_sign}/{len(results)} "
          f"({100*same_sign/len(results):.1f}%) — predicted and actual "
          f"drops have the same direction.")
