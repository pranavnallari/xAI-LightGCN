# main.py — unified train / eval / explain entry point

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from model import LightGCN
from data import load_dataset, preprocess_data, temporal_split
from graph import build_normalized_adj
from metrics import evaluate_lightgcn
from explainer import LightGCNExplainer

# -----------------------------
# Config
# -----------------------------
EMBEDDING_DIM = 64
NUM_LAYERS = 3
LR = 1e-3
EPOCHS = 200
BATCH_SIZE = 4096
LAMBDA_REG = 1e-6

MODEL_PATH = "best_model.pth"
FINAL_CKPT = "lightgcn_final.pth"

DATA_FILES = ["train.pt", "val.pt", "test.pt"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}")

# -----------------------------
# Data loading (shared)
# -----------------------------
def load_data():
    ratings_df = load_dataset("../dataset/ratings.dat")
    ratings_df, user2idx, _, movie2idx, _ = preprocess_data(ratings_df)
    train_int, val_int, test_int = temporal_split(ratings_df)

    return (
        train_int.to(device),
        val_int.to(device),
        test_int.to(device),
        len(user2idx),
        len(movie2idx),
    )

# -----------------------------
# Positives map
# -----------------------------
def build_user_pos_items(train_int):
    user_pos = {}
    for u, i in train_int.cpu().tolist():
        user_pos.setdefault(u, set()).add(i)
    return user_pos

# -----------------------------
# Model builder
# -----------------------------
def build_model(num_users, num_items, train_int):
    norm_adj = build_normalized_adj(num_users, num_items, train_int, device=device)
    return LightGCN(
        num_users, num_items, EMBEDDING_DIM, NUM_LAYERS, norm_adj, device
    ).to(device)

# -----------------------------
# Training
# -----------------------------
def train_model(model, train_int, val_int, num_items, user_pos_items):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

    num_edges = train_int.size(0)
    best_ndcg = 0.0

    print("\n🔥 Training started")

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(num_edges, device=device)

        epoch_loss = 0.0
        pbar = tqdm(range(0, num_edges, BATCH_SIZE), leave=False)

        for i in pbar:
            user_emb, item_emb = model.get_embeddings()

            idx = perm[i : i + BATCH_SIZE]
            users = train_int[idx, 0]
            pos_items = train_int[idx, 1]

            neg_items = torch.randint(0, num_items, (users.size(0),), device=device)
            for j, u in enumerate(users.tolist()):
                while neg_items[j].item() in user_pos_items[u]:
                    neg_items[j] = torch.randint(0, num_items, (1,), device=device)

            pos_scores, neg_scores = model(
                users, pos_items, neg_items, user_emb, item_emb
            )

            bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
            reg = (
                model.user_embeddings.weight.pow(2).sum()
                + model.item_embeddings.weight.pow(2).sum()
            )
            loss = bpr + LAMBDA_REG * reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        scheduler.step()
        print(f"📈 Epoch {epoch+1:03d} | Loss {epoch_loss/num_edges:.4f}")

        if (epoch + 1) % 100 == 0:
            model.eval()
            res = evaluate_lightgcn(model, train_int, val_int, num_items)
            ndcg = res["NDCG@10"]

            print(f"  Val NDCG@10: {ndcg:.4f}")
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                torch.save(model.state_dict(), MODEL_PATH)
                print("  ⭐ Saved best model")

    print("✅ Training complete")

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, train_int, test_int, num_items):
    print("\n🏁 Running evaluation")
    res = evaluate_lightgcn(model, train_int, test_int, num_items)
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")
    return res

# -----------------------------
# Explainability
# -----------------------------
def run_explainability(model, user_pos_items):
    print("\n🧠 Running explainability")

    explainer = LightGCNExplainer(model, user_pos_items)

    u = max(user_pos_items.items(), key=lambda x: len(x[1]))[0]

    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings()
        scores = user_emb[u] @ item_emb.T
        scores[list(user_pos_items[u])] = -1e9
        i = torch.topk(scores, 1).indices.item()

    exp = explainer.explain(u, i, max_hops=3)

    print(f"\n🔍 User {u} → Item {i}")
    print(f"  Dominant layer: {exp['k_star']}")
    print("  Contributions:", exp["contributions"])
    print("  Paths:")
    for p in exp["paths"]:
        print("   ", p)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    train_int, val_int, test_int, num_users, num_items = load_data()
    user_pos_items = build_user_pos_items(train_int)

    model = build_model(num_users, num_items, train_int)

    if os.path.exists(MODEL_PATH):
        print(f"✅ Found pretrained model: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        train_model(model, train_int, val_int, num_items, user_pos_items)

    test_res = evaluate(model, train_int, test_int, num_items)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": test_res,
            "train_int": train_int.cpu(),
            "test_int": test_int.cpu(),
        },
        FINAL_CKPT,
    )

    run_explainability(model, user_pos_items)
    print("\n🎉 Done!")
