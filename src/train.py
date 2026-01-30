from model import LightGCN
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm  # ✅ Progress bars
import os
from data import load_dataset, preprocess_data, temporal_split
from graph import build_normalized_adj
from eval import evaluate_lightgcn

# -----------------------------
# Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}")

embedding_dim, num_layers, learning_rate = 64, 3, 1e-3
num_epochs, batch_size, lambda_reg = 200, 4096, 1e-6

# -----------------------------
# Data ✅ Temporal
# -----------------------------
print("📊 Loading data...")
ratings_df = load_dataset("../dataset/ratings.dat")
ratings_df, user2idx, idx2user, movie2idx, idx2movie = preprocess_data(ratings_df)
train_int, val_int, test_int = temporal_split(ratings_df)

print(f"  Train: {train_int.size(0):,} edges")
print(f"  Val:   {val_int.size(0):,} edges")
print(f"  Test:  {test_int.size(0):,} edges")

torch.save(train_int.cpu(), "train.pt")
torch.save(val_int.cpu(), "val.pt")
torch.save(test_int.cpu(), "test.pt")

num_users = len(user2idx)
num_items = len(movie2idx)
train_int = train_int.to(device)

print(f"👥 Users: {num_users:,} | 🎬 Items: {num_items:,}")

# Positives map
user_pos_items = {}
for u, i in train_int.cpu().tolist():
    user_pos_items.setdefault(u, set()).add(i)

# Graph
print("🔗 Building normalized adj...")
norm_adj = build_normalized_adj(num_users, num_items, train_int, device=device)

# Model
model = LightGCN(num_users, num_items, embedding_dim, num_layers, norm_adj, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

print("Training...")

# -----------------------------
# Logging vars
# -----------------------------
num_edges = train_int.size(0)
best_ndcg = 0
peak_mem = 0

for epoch in range(num_epochs):
    model.train()
    perm = torch.randperm(num_edges, device=device)
    epoch_loss = 0.0
    num_batches = 0

    # Progress bar per epoch
    epoch_pbar = tqdm(range(0, num_edges, batch_size), desc=f"Epoch {epoch+1:04d}", leave=False)

    for i in epoch_pbar:
        user_emb_all, item_emb_all = model.get_embeddings()
        idx = perm[i:i + batch_size]
        users = train_int[idx, 0]
        pos_items = train_int[idx, 1]

        # Negatives
        neg_items = torch.randint(0, num_items, (users.size(0),), device=device)
        for j, u in enumerate(users.tolist()):
            while neg_items[j].item() in user_pos_items.get(u, set()):
                neg_items[j] = torch.randint(0, num_items, (1,), device=device)

        pos_scores, neg_scores = model(users, pos_items, neg_items, user_emb_all, item_emb_all)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        reg_loss = (model.user_embeddings.weight.pow(2).sum() +
                    model.item_embeddings.weight.pow(2).sum())
        loss = bpr_loss + lambda_reg * reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss * idx.size(0)
        num_batches += 1

        # Live batch avg
        epoch_pbar.set_postfix({"Loss": f"{batch_loss:.3f}"})

    # Epoch summary
    avg_loss = epoch_loss / num_edges
    scheduler.step()
    lr = scheduler.get_last_lr()[0]

    print(f"📈 Epoch {epoch+1:04d} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
          f"Mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Val every 100 + peak mem
    if (epoch + 1) % 100 == 0:
        model.eval()
        val_res = evaluate_lightgcn(model, train_int, val_int.to(device), num_items, k=10)
        ndcg = val_res['NDCG@10']
        print(f"  Val → Hit@10: {val_res['Hit@10']:.4f} | NDCG@10: {ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ⭐ New best NDCG: {ndcg:.4f} → Saved!")


        # Mem log
        mem_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        peak_mem = max(peak_mem, mem_gb)
        print(f"  Peak GPU mem: {peak_mem:.1f}GB")
        model.train()

# Final test
print("🏁 Final test eval...")
test_res = evaluate_lightgcn(model, train_int, test_int.to(device), num_items, k=10)

print("\n🎯 LightGCN Test Results:")
for metric, score in test_res.items():
    print(f"  {metric}: {score:.4f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'user_emb': model.get_embeddings()[0].cpu(),
    'item_emb': model.get_embeddings()[1].cpu(),
    'train_int': train_int.cpu(),
    'val_int': val_int.cpu(),
    'test_int': test_int.cpu(),
    'metrics': test_res
}, "lightgcn_final.pth")
print("✅ Complete run saved!")
