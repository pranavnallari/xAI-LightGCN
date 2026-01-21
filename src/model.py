# model.py
import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, norm_adj, device):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device
        self.register_buffer('norm_adj', norm_adj.to(device))
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.01)                      # Initialize user embeddings with a normal distribution (mean=0, std=0.01) for small random values
        nn.init.normal_(self.item_embeddings.weight, std=0.01)                      # Initialize item embeddings with a normal distribution (mean=0, std=0.01) for small random values

    def forward(self, users, pos_items, neg_items, user_emb_all, item_emb_all):
        user_emb = user_emb_all[users]
        pos_item_emb = item_emb_all[pos_items]
        neg_item_emb = item_emb_all[neg_items]

        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=-1)

        return pos_scores, neg_scores


    def get_embeddings(self):
        all_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        ego_embeddings = all_embeddings
        for _ in range(self.num_layers):
            all_embeddings = torch.spmm(self.norm_adj, all_embeddings)
            ego_embeddings += all_embeddings
        final_embeddings = ego_embeddings / (self.num_layers + 1)
        user_emb = final_embeddings[:self.num_users]
        item_emb = final_embeddings[self.num_users:]
        return user_emb, item_emb
