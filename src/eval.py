# eval.py - Check for model → eval OR train from scratch
from model import LightGCN
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from data import load_dataset, preprocess_data, temporal_split
from graph import build_normalized_adj
from metrics import evaluate_lightgcn
from explainer import LightGCNExplainer
# -----------------------------
# Model checkpoint check
# -----------------------------
MODEL_FILES = ["lightgcn_final.pth", "best_model.pth", "lightgcn_model.pth"]
TRAIN_FILES = ["train.pt", "val.pt", "test.pt"]

def load_pretrained_model(model_file, num_users, num_items, embedding_dim=64, num_layers=3):
    """Load pretrained model if exists"""
    if not os.path.exists(model_file):
        return None

    print(f"✅ Found pretrained model: {model_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build minimal graph for embedding computation
    ratings_df = load_dataset("../dataset/ratings.dat")
    ratings_df, user2idx, idx2user, movie2idx, idx2movie = preprocess_data(ratings_df)
    train_int, _, _ = temporal_split(ratings_df)
    num_users, num_items = len(user2idx), len(movie2idx)
    train_int = train_int.to(device)

    norm_adj = build_normalized_adj(num_users, num_items, train_int, device=device)

    model = LightGCN(num_users, num_items, embedding_dim, num_layers, norm_adj, device).to(device)
    checkpoint = torch.load(model_file, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"✅ Loaded {model_file} successfully!")
    return model, train_int, checkpoint.get('test_int', None), num_items

# -----------------------------
# MAIN LOGIC
# -----------------------------
print("🔍 Checking for pretrained models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {device}")

# Check if we can skip training
model = None
loaded_model_file = None
for model_file in MODEL_FILES:
    model, train_int, test_int, num_items = load_pretrained_model(model_file, 6040, 3706)
    if model is not None:
        loaded_model_file = model_file
        break

if model is not None:
    print("\n🎯 EVALUATION MODE - Using pretrained model")
    print("="*60)

    # Load test data if not in checkpoint
    if test_int is None or not test_int.is_cuda:
        ratings_df = load_dataset("../dataset/ratings.dat")
        ratings_df, _, _, _, _ = preprocess_data(ratings_df)
        _, _, test_int = temporal_split(ratings_df)
        test_int = test_int.to(device)

    # Evaluate
    print("🏁 Running full test evaluation...")
    test_results = evaluate_lightgcn(model, train_int, test_int, num_items, k=10)

    print("\n🎯 FINAL LightGCN Test Results:")
    print("="*40)
    for metric, score in test_results.items():
        print(f"  {metric}: {score:.4f}")

    print(f"\n✅ Results from: {loaded_model_file}")
print("\n🎉 Done!")




# Create explainer
explainer = LightGCNExplainer(
    model=model,
    user_pos_items=user_pos_items
)

# 1️⃣ Pick a high-activity user (most interactions)
u = max(user_pos_items.items(), key=lambda x: len(x[1]))[0]

# 2️⃣ Pick a top-ranked recommended item (not interacted)
with torch.no_grad():
    user_emb, item_emb = model.get_embeddings()
    scores = user_emb[u] @ item_emb.T
    scores[list(user_pos_items[u])] = -1e9  # mask seen items
    i = torch.topk(scores, 1).indices.item()

# 3️⃣ Run explainer
explanation = explainer.explain(
    user_id=u,
    item_id=i,
    max_hops=3
)

# 4️⃣ Print results (FIXED variables)
print(f"\n🔍 Explanation for User {u} → Item {i}")
print(f"  Dominant layer (k*): {explanation['k_star']}")

print("  Layer contributions:")
if explanation["contributions"] is not None:
    for k, c in enumerate(explanation["contributions"]):
        print(f"    Layer {k}: {c:.3f}")
else:
    print("    (No meaningful contribution)")

print("  Propagation paths:")
if explanation["paths"]:
    for path in explanation["paths"]:
        print("   ", path)
else:
    print("    (No graph-based paths — identity-driven recommendation)")
