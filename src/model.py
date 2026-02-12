# model.py
import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, norm_adj, device):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device

        self.register_buffer("norm_adj", norm_adj.to(device))

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    # ---------- PURE BASELINE PROPAGATION ----------
    def get_embeddings(self, return_layers=False):
        all_embeddings = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0
        )

        embeddings_per_layer = [all_embeddings] if return_layers else None

        final_embeddings = all_embeddings

        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            final_embeddings += all_embeddings

            if return_layers:
                embeddings_per_layer.append(all_embeddings)

        final_embeddings = final_embeddings / (self.num_layers + 1)

        user_final = final_embeddings[:self.num_users]
        item_final = final_embeddings[self.num_users:]

        if return_layers:
            return user_final, item_final, embeddings_per_layer

        return user_final, item_final

    # ---------- STANDARD BPR FORWARD ----------
    def forward(self, users, pos_items, neg_items, user_emb_all, item_emb_all):
        user_emb = user_emb_all[users]
        pos_item_emb = item_emb_all[pos_items]
        neg_item_emb = item_emb_all[neg_items]

        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=-1)

        return pos_scores, neg_scores
