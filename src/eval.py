# eval.py
# metrics
# eval.py - PHASE 1 ONLY: NDCG@10, Recall@10, Precision@10
import torch
import numpy as np
from collections import defaultdict

def hit_rate(pred_items, true_items, k=10):
    """Hit Rate@K"""
    return int(true_items[0] in pred_items[:k])

def precision_at_k(pred_items, true_items, k=10):
    """Precision@K"""
    hits = hit_rate(pred_items, true_items, k)
    return hits / k

def recall_at_k(pred_items, true_items, k=10):
    """Recall@K (leave-one-out: true_items has 1 item)"""
    hits = hit_rate(pred_items, true_items, k)
    return hits / len(true_items)

def ndcg_at_k(scores, true_item, k=10):
    """NDCG@K (simplified for single true item)"""
    pred_scores = scores[:k]
    true_scores = torch.zeros(k)
    true_pos = torch.argmin(torch.abs(pred_scores - scores[true_item]))
    true_scores[true_pos] = 1.0

    dcg = (true_scores / np.log2(np.arange(2, k+2))).sum()
    idcg = (torch.ones(k) / np.log2(np.arange(2, k+2))).sum()
    return dcg / idcg if idcg > 0 else 0

def evaluate_baseline(model, test_interactions, num_items, num_users, k=10):
    """Evaluate LightGCN baseline on test set"""
    model.eval()
    metrics = {"Precision@10": [], "Recall@10": [], "NDCG@10": [], "Hit@10": []}

    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings()

        for user_id, true_item in test_interactions:
            # Score all items
            scores = torch.mv(item_emb, user_emb[user_id])

            # Top-K
            topk_scores, topk_items = torch.topk(scores, k)

            # Metrics
            metrics["Hit@10"].append(hit_rate(topk_items, [true_item]))
            metrics["Precision@10"].append(precision_at_k(topk_items, [true_item]))
            metrics["Recall@10"].append(recall_at_k(topk_items, [true_item]))
            metrics["NDCG@10"].append(ndcg_at_k(scores, true_item))

    # Averages
    results = {k: np.mean(v) for k, v in metrics.items()}
    return results

# USAGE
if __name__ == "__main__":
    from data import load_dataset, preprocess_data, train_test_split_per_user
    from graph import build_normalized_adj
    from model import LightGCN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    ratings_df = load_dataset("../dataset/ratings.dat")
    ratings_df, user2idx, _, movie2idx, _ = preprocess_data(ratings_df)
    _, test_int = train_test_split_per_user(ratings_df)
    num_users, num_items = len(user2idx), len(movie2idx)

    # Load model
    train_int, _ = train_test_split_per_user(ratings_df)
    norm_adj = build_normalized_adj(num_users, num_items, train_int.to(device), device)
    model = LightGCN(num_users, num_items, 64, 3, norm_adj, device)
    model.load_state_dict(torch.load("lightgcn_model_two.pth"))
    model.to(device)

    # Evaluate
    results = evaluate_baseline(model, test_int, num_items, num_users)
    print("LightGCN Baseline Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
