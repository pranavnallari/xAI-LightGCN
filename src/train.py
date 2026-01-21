# train.py
# Training script for LightGCN model

from model import LightGCN
import torch.optim as optim
import torch
import torch.nn.functional as F
from data import load_dataset, preprocess_data, train_test_split_per_user
from graph import build_normalized_adj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device : {device}")

embedding_dim = 64
num_layers = 3
learning_rate = 1e-3
num_epochs = 1000
lambda_reg = 1e-6

ratings_df = load_dataset("../dataset/ratings.dat")
ratings_df, user2idx, idx2user, movie2idx, idx2movie = preprocess_data(ratings_df)
train_interactions, test_interactions = train_test_split_per_user(ratings_df)

num_users = len(user2idx)
num_items = len(movie2idx)

print(f"Users: {num_users}, Items: {num_items}")
print(f"Train: {len(train_interactions)}, Test: {len(test_interactions)}")

norm_adj = build_normalized_adj(num_users, num_items, train_interactions.to(device), device=device)

model = LightGCN(num_users, num_items, embedding_dim, num_layers, norm_adj, device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training...")
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    users = train_interactions[:, 0].to(device)
    pos_items = train_interactions[:, 1].to(device)
    neg_items = torch.randint(0, num_items, (len(train_interactions),), device=device)

    # Forward pass
    pos_scores, neg_scores = model(users, pos_items, neg_items)
    logits = pos_scores - neg_scores  # [batch, batch]
    bpr_loss = -torch.log(torch.sigmoid(logits)).mean()

    # L2 Regularization
    user_embeddings = model.user_embeddings.weight
    item_embeddings = model.item_embeddings.weight

    # Calculate L2 norm of embeddings (squared sum of elements)
    reg_loss = (user_embeddings**2).sum() + (item_embeddings**2).sum()

    # Combine BPR loss and regularization term with a weight factor (lambda)
    loss = bpr_loss + lambda_reg * reg_loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    total_loss = loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}')


torch.save(model.state_dict(), "lightgcn_model_one.pth")
print("✅ Model saved: lightgcn_model.pth")
