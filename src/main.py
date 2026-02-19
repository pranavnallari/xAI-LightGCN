# main.py

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm

from data import load_movielens_1m, load_movie_metadata
from graph import build_adj
from model import LightGCN
from evaluation import run_explainability_tests
from counterfactual import run_counterfactual_tests
from explainer import LightGCNExplainer
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
MOVIES_FILE = "../dataset/movies.dat"

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
movie_names, movie_genres = load_movie_metadata(MOVIES_FILE)

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
# Explainability Tests
# ------------------------------------------------
# run_explainability_tests(
#     model,
#     train_dict,
#     movie_names,
#     movie_genres,
#     test_data,
#     num_cases=5,
#     top_k=5
# )

explainer = LightGCNExplainer(model, train_dict, movie_names, movie_genres)

run_counterfactual_tests(
    model,
    train_dict,
    explainer,
    num_users=10,
    top_k=5
)
