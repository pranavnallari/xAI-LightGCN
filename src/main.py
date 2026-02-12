# main.py

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm

from explainer import LightGCNExplainer
from data import load_movielens_1m
from graph import build_adj
from model import LightGCN
from metrics import compute_metrics


# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# -----------------------------
# Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RATINGS_FILE = "../dataset/ratings.dat"
EMBEDDING_DIM = 64
N_LAYERS = 3
LR = 0.001
EPOCHS = 100
BATCH_SIZE = 2048
TOP_K = 10
MODEL_PATH = "lightgcn_100epoch.pth"


# -----------------------------
# Load Data
# -----------------------------
train_data, val_data, test_data, num_users, num_items = \
    load_movielens_1m(RATINGS_FILE)

train_dict = {}
for u, i in train_data:
    train_dict.setdefault(u, set()).add(i)

adj = build_adj(num_users, num_items, train_data, device)

model = LightGCN(num_users, num_items, EMBEDDING_DIM, N_LAYERS, adj).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)


# ------------------------------------------------
# Training or Load
# ------------------------------------------------
if os.path.exists(MODEL_PATH):

    print("\nLoading saved model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

else:

    print("\nTraining model...")

    def generate_batches(data, batch_size):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs"):

        model.train()
        epoch_loss = 0

        for batch in generate_batches(train_data, BATCH_SIZE):

            optimizer.zero_grad()

            users, pos_items = zip(*batch)
            users = torch.LongTensor(users).to(device)
            pos_items = torch.LongTensor(pos_items).to(device)

            user_emb, item_emb = model()

            neg_items = []
            for u in users:
                uid = u.item()
                while True:
                    neg = random.randrange(num_items)
                    if neg not in train_dict[uid]:
                        neg_items.append(neg)
                        break

            neg_items = torch.LongTensor(neg_items).to(device)

            u_emb = user_emb[users]
            pos_emb = item_emb[pos_items]
            neg_emb = item_emb[neg_items]

            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)

            loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch)

        epoch_loss /= len(train_data)
        print(f"Epoch {epoch} | Avg Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("\nModel saved successfully.\n")


# ------------------------------------------------
# Evaluation
# ------------------------------------------------
print("\n--- Evaluation ---")

model.eval()

hits, precisions, ndcgs, mrrs = [], [], [], []

with torch.no_grad():

    user_emb, item_emb = model()

    ground_truth = {}
    for u, i in test_data:
        ground_truth.setdefault(u, []).append(i)

    for user, true_items in ground_truth.items():

        scores = torch.matmul(item_emb, user_emb[user])

        if user in train_dict:
            scores[list(train_dict[user])] = -1e9

        top_items = torch.topk(scores, k=TOP_K).indices.cpu().tolist()

        hit, prec, ndcg, rr = compute_metrics(top_items, true_items, TOP_K)

        hits.append(hit)
        precisions.append(prec)
        ndcgs.append(ndcg)
        mrrs.append(rr)

print(f"Hit@{TOP_K}: {np.mean(hits):.4f}")
print(f"Precision@{TOP_K}: {np.mean(precisions):.4f}")
print(f"NDCG@{TOP_K}: {np.mean(ndcgs):.4f}")
print(f"MRR@{TOP_K}: {np.mean(mrrs):.4f}")


# ------------------------------------------------
# Explainability Test Case
# ------------------------------------------------
print("\n--- Explainability Test Case ---")

explainer = LightGCNExplainer(model, train_dict)

test_user = list(train_dict.keys())[0]

with torch.no_grad():
    user_emb, item_emb = model()
    test_scores = torch.matmul(item_emb, user_emb[test_user])
    test_scores[list(train_dict[test_user])] = -1e9
    test_top_item = torch.topk(test_scores, k=1).indices.item()

print(f"User: {test_user}")
print(f"Top recommended item: {test_top_item}")

explanation = explainer.explain(test_user, test_top_item, top_k=5)

print("\nLayer-wise contributions:")
for k, val in enumerate(explanation["layer_scores"]):
    print(f"Layer {k}: {val:.6f}")

print(f"\nDominant depth (k*): {explanation['dominant_depth']}")

if explanation["top_training_contributors"] is not None:
    print("\nTop contributing training items (k=1 case):")
    for item_id, score in explanation["top_training_contributors"]:
        print(f"Item {item_id} | Contribution: {score:.6f}")


# ------------------------------------------------
# Global Layer Dominance Analysis
# ------------------------------------------------
print("\n--- Global Layer Dominance Analysis ---")

num_samples = 500
users = list(train_dict.keys())[:num_samples]

layer_counts = [0] * (model.n_layers + 1)
layer_magnitudes = [0.0] * (model.n_layers + 1)

with torch.no_grad():

    user_emb, item_emb = model()

    for user in users:

        scores = torch.matmul(item_emb, user_emb[user])
        scores[list(train_dict[user])] = -1e9

        loop_top_item = torch.topk(scores, k=1).indices.item()

        layer_scores, _ = explainer.exact_layer_contributions(user, loop_top_item)
        abs_scores = [abs(s) for s in layer_scores]

        k_star = int(np.argmax(abs_scores))
        layer_counts[k_star] += 1

        for k in range(len(abs_scores)):
            layer_magnitudes[k] += abs_scores[k]

layer_magnitudes = [m / num_samples for m in layer_magnitudes]

print("\nDominant Depth Distribution:")
for k, count in enumerate(layer_counts):
    print(f"Layer {k}: {count / num_samples:.3f}")

print("\nAverage Absolute Contribution Per Layer:")
for k, mag in enumerate(layer_magnitudes):
    print(f"Layer {k}: {mag:.4f}")


# ------------------------------------------------
# Final Generated Explanation
# ------------------------------------------------
print("\nGenerated Explanation:")
print(explainer.generate_explanation(test_user, test_top_item))



# ------------------------------------------------
# Multiple Explainability Test Cases
# ------------------------------------------------
print("\n--- Multiple Explainability Test Cases ---")

sample_users = random.sample(list(train_dict.keys()), 3)

with torch.no_grad():
    user_emb, item_emb = model()

    for idx, u in enumerate(sample_users):

        scores = torch.matmul(item_emb, user_emb[u])
        scores[list(train_dict[u])] = -1e9
        recommended_item = torch.topk(scores, k=1).indices.item()

        print(f"\nCase {idx+1}")
        print(f"User: {u}")
        print(f"Top recommended item: {recommended_item}")

        layer_scores, _ = explainer.exact_layer_contributions(u, recommended_item)

        print("Layer contributions:")
        for k, val in enumerate(layer_scores):
            print(f"  Layer {k}: {val:.6f}")

        print("Dominant depth:", np.argmax(np.abs(layer_scores)))

        print("Explanation:")
        print(" ", explainer.generate_explanation(u, recommended_item))
