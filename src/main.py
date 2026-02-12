# main.py

import os
import random
import argparse
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from model import LightGCN
from data import load_dataset, preprocess_data, temporal_split
from graph import build_normalized_adj
from metrics import evaluate_lightgcn

# -----------------------------
# Config – **FINAL TUNED SETTINGS**
# -----------------------------
EMBEDDING_DIM = 64
NUM_LAYERS = 3
LR = 1e-3
EPOCHS = 300
BATCH_SIZE = 4096
LAMBDA_REG = 5e-4
VALIDATION_FREQ = 5
MAX_NEG_ATTEMPTS = 100
GRADIENT_CLIP = 1.0
BPR_TEMPERATURE = 0.2

MODEL_PATH = "best_model.pth"
FINAL_CKPT = "lightgcn_final.pth"

# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LightGCN Training")
    parser.add_argument("--data_path", type=str, default="../dataset/ratings.dat",
                        help="Path to ratings.dat file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------
# Data loading (with mappings)
# -----------------------------
def load_data(data_path):
    ratings_df = load_dataset(data_path)
    ratings_df, user2idx, idx2user, movie2idx, idx2movie = preprocess_data(ratings_df)
    train_int, val_int, test_int = temporal_split(ratings_df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        train_int.to(device),
        val_int.to(device),
        test_int.to(device),
        len(user2idx),
        len(movie2idx),
        user2idx,
        movie2idx,
        idx2user,
        idx2movie,
    )

# -----------------------------
# Positive items map (CPU set)
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
        num_users, num_items, EMBEDDING_DIM, NUM_LAYERS, norm_adj
    ).to(device)

# -----------------------------
# BPR loss with temperature scaling – 🆕 FIXES SATURATION
# -----------------------------
def bpr_loss(pos_scores, neg_scores, tau=BPR_TEMPERATURE):
    return -torch.nn.functional.logsigmoid((pos_scores - neg_scores) / tau).mean()

# -----------------------------
# Training
# -----------------------------
def train_model(model, train_int, val_int, num_items, user_pos_items):
    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,          # Allow more epochs before reducing LR
        threshold=1e-4,
        cooldown=2
    )

    num_edges = train_int.size(0)
    best_ndcg = 0.0
    epochs_no_improve = 0
    early_stop_patience = 20

    print("\n🔥 Training started")

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(num_edges, device=device)
        epoch_loss = 0.0

        pbar = tqdm(range(0, num_edges, BATCH_SIZE), desc=f"Epoch {epoch+1}", leave=False)

        for i in pbar:
            user_emb_all, item_emb_all = model.get_embeddings(verbose=False)

            idx = perm[i:i + BATCH_SIZE]
            users = train_int[idx, 0]
            pos_items = train_int[idx, 1]

            # Negative sampling
            neg_items = torch.randint(0, num_items, (users.size(0),), device=device)
            for j, u in enumerate(users.tolist()):
                attempts = 0
                while neg_items[j].item() in user_pos_items[u] and attempts < MAX_NEG_ATTEMPTS:
                    neg_items[j] = torch.randint(0, num_items, (1,), device=device)
                    attempts += 1

            # Forward pass
            pos_scores, neg_scores = model.forward_score(
                users, pos_items, neg_items, user_emb_all, item_emb_all
            )

            loss_bpr = bpr_loss(pos_scores, neg_scores, tau=BPR_TEMPERATURE)

            # L2 regularization on FINAL embeddings
            reg_loss = (
                user_emb_all[:model.num_users].pow(2).sum() +
                item_emb_all[model.num_users:].pow(2).sum()
            ) * LAMBDA_REG

            loss = loss_bpr + reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

            epoch_loss += loss_bpr.item() * users.size(0)
            pbar.set_postfix({
                "bpr": f"{loss_bpr.item():.3f}",
                "reg": f"{reg_loss.item():.3f}"
            })

        avg_loss = epoch_loss / num_edges
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📈 Epoch {epoch+1:03d} | BPR Loss {avg_loss:.4f} | LR {current_lr:.2e}")

        # -----------------------------
        # Validation
        # -----------------------------
        if (epoch + 1) % VALIDATION_FREQ == 0 or epoch + 1 == EPOCHS:
            model.eval()

            val_res = evaluate_lightgcn(
                model,
                train_int,
                val_int.to(device),
                num_items,
                k=10,
                use_full_matrix=False,   # Memory efficient
                verbose=False
            )
            ndcg = val_res["NDCG@10"]
            recall = val_res["Recall@10"]
            hit = val_res["Hit@10"]
            print(f"  Val NDCG@10: {ndcg:.4f} | Recall@10: {recall:.4f} | Hit@10: {hit:.4f}")

            # Training NDCG (subset) – detect overfitting
            train_res = evaluate_lightgcn(
                model,
                train_int,
                train_int[:10000],      # Quick check
                num_items,
                k=10,
                use_full_matrix=False,
                verbose=False
            )
            train_ndcg = train_res["NDCG@10"]
            print(f"  Train NDCG@10 (subset): {train_ndcg:.4f}")

            scheduler.step(ndcg)

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_PATH)
                print("  ⭐ Saved best model")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    print(f"⏹️ Early stopping after {epoch+1} epochs")
                    break

    print("✅ Training complete")

# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, train_int, test_int, num_items, split_name="Test"):
    print(f"\n🏁 Running {split_name} evaluation")
    res = evaluate_lightgcn(
        model,
        train_int,
        test_int,
        num_items,
        k=10,
        use_full_matrix=False,
        verbose=True
    )
    for k, v in res.items():
        print(f"  {k}: {v:.4f}")
    return res

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")
    print(f"🎲 Random Seed: {args.seed}")

    train_int, val_int, test_int, num_users, num_items, user2idx, movie2idx, idx2user, idx2movie = load_data(args.data_path)

    user_pos_items = build_user_pos_items(train_int)

    model = build_model(num_users, num_items, train_int)

    if os.path.exists(MODEL_PATH):
        print(f"✅ Found pretrained model: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        train_model(model, train_int, val_int, num_items, user_pos_items)
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    test_res = evaluate(model, train_int, test_int, num_items, "Test")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": test_res,
            "train_int": train_int.cpu(),
            "val_int": val_int.cpu(),
            "test_int": test_int.cpu(),
            "user2idx": user2idx,
            "movie2idx": movie2idx,
            "idx2user": idx2user,
            "idx2movie": idx2movie,
            "config": {
                "embedding_dim": EMBEDDING_DIM,
                "num_layers": NUM_LAYERS,
                "lr": LR,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lambda_reg": LAMBDA_REG,
                "bpr_temperature": BPR_TEMPERATURE,
                "seed": args.seed,
            },
        },
        FINAL_CKPT,
    )
    print(f"\n🎉 Done! Final checkpoint saved to {FINAL_CKPT}")
