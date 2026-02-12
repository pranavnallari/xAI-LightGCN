# model.py

import torch


class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_items, dim, n_layers, adj):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        self.adj = adj

        self.embedding = torch.nn.Embedding(num_users + num_items, dim)
        torch.nn.init.normal_(self.embedding.weight, std=0.1)

    def propagate(self):

        all_emb = self.embedding.weight
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.adj, all_emb)
            embs.append(all_emb)

        return embs

    def forward(self):

        embs = self.propagate()
        final_emb = torch.mean(torch.stack(embs, dim=0), dim=0)

        user_emb = final_emb[:self.num_users]
        item_emb = final_emb[self.num_users:]

        return user_emb, item_emb
