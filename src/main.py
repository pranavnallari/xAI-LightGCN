# main.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import logging
from tqdm import tqdm


from explainer import LightGCNExplainer
from data import load_movielens_1m
from graph import build_adj
from model import LightGCN
from metrics import compute_metrics


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


# -----------------------------
# Training
# -----------------------------
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


# -----------------------------
# Evaluation
# -----------------------------
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

# Pick a user with test interactions
test_user = list(train_dict.keys())[0]

with torch.no_grad():
    user_emb, item_emb = model()

    scores = torch.matmul(item_emb, user_emb[test_user])

    # Mask training items
    scores[list(train_dict[test_user])] = -1e9

    top_item = torch.topk(scores, k=1).indices.item()

print(f"User: {test_user}")
print(f"Top recommended item: {top_item}")

explanation = explainer.explain(test_user, top_item, top_k=5)

print("\nLayer-wise contributions:")
for k, val in enumerate(explanation["layer_scores"]):
    print(f"Layer {k}: {val:.6f}")

print(f"\nDominant depth (k*): {explanation['dominant_depth']}")

if explanation["top_training_contributors"] is not None:
    print("\nTop contributing training items (k=1 case):")
    for item_id, score in explanation["top_training_contributors"]:
        print(f"Item {item_id} | Contribution: {score:.6f}")
