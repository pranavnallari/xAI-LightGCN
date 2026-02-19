# evaluation.py

import torch
import numpy as np
import random

from explainer import LightGCNExplainer


# ------------------------------------------------
# Pretty Table
# ------------------------------------------------
def print_item_table(item_ids, explainer, title="Items"):
    print(f"\n{title}")
    print(f"{'Movie Name':40} | {'ID':6} | Categories")
    print("-" * 90)

    for item_id in item_ids:
        name = explainer.movie_names.get(item_id, "Unknown")
        genres = explainer.movie_genres.get(item_id, "")
        print(f"{name[:40]:40} | {item_id:6} | {genres}")


# ------------------------------------------------
# Explainability Tests
# ------------------------------------------------
def run_explainability_tests(
    model,
    train_dict,
    movie_names,
    movie_genres,
    test_data,
    num_cases=5,
    top_k=5,
    verbose=True,
):
    """
    Runs clean explainability diagnostics.
    """

    if verbose:
        print("\n--- Explainability Tests ---")

    model.eval()
    explainer = LightGCNExplainer(model, train_dict, movie_names, movie_genres)

    results = []

    with torch.no_grad():
        user_emb, item_emb = model()

        def top_item(user):
            if user not in train_dict:
                return None

            scores = torch.matmul(item_emb, user_emb[user])
            scores[list(train_dict[user])] = -1e9
            return torch.topk(scores, k=1).indices.item()

        # pick valid users
        users = [u for u, _ in test_data if u in train_dict]
        users = list(set(users))

        if not users:
            print("No valid users for explainability.")
            return []

        users = random.sample(users, min(num_cases, len(users)))

        for idx, u in enumerate(users, 1):

            rec = top_item(u)
            if rec is None:
                continue
            layer_scores, _ = explainer.exact_layer_contributions(u, rec)
            k_star = int(np.argmax(np.abs(layer_scores)))

            contribs = explainer.one_hop_contributions(u, rec, top_k)

            # Pass pre-computed layer_scores to avoid recomputing
            explanation_text = explainer.generate_explanation(
                u, rec, top_k, _layer_scores=layer_scores
            )

            results.append({
                "user": u,
                "item": rec,
                "k_star": k_star,
                "layer_scores": layer_scores,
                "contributors": contribs,
                "explanation": explanation_text,
            })

            # ---------- Printing ----------
            if verbose:
                print(f"\n==============================")
                print(f"Case {idx}")
                print(f"User: {u}")
                print(f"Recommended: {explainer.fmt(rec)}")

                print("\nLayer contributions:")
                for k, val in enumerate(layer_scores):
                    print(f"  Layer {k}: {val:.4f}")

                print("\nDominant depth:", k_star)

                # Contributor table
                if contribs:
                    ids = [item_id for item_id, _ in contribs]
                    print_item_table(ids, explainer, "Top Contributing Items")

                print("\nExplanation:")
                print(" ", explanation_text)

    return results
