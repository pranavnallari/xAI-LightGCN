# model.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        norm_adj: torch.Tensor,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.register_buffer("norm_adj", norm_adj)

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embeddings.weight, std=0.1)
        nn.init.normal_(self.item_embeddings.weight, std=0.1)

    def get_embeddings(
        self, return_layers: bool = False, verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        all_embeddings = torch.cat(
            [self.user_embeddings.weight, self.item_embeddings.weight], dim=0
        )

        if verbose:
            print(f"\n🔬 Layer0 item norm: {all_embeddings[self.num_users:].norm().item():.3f}")

        embeddings_per_layer = [all_embeddings] if return_layers else None
        final_embeddings = all_embeddings.clone()

        for layer in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            final_embeddings += all_embeddings

            if verbose:
                print(f"  Layer{layer+1} item norm: {all_embeddings[self.num_users:].norm().item():.3f}")

            if return_layers:
                embeddings_per_layer.append(all_embeddings)

        final_embeddings = final_embeddings / (self.num_layers + 1)
        user_final = final_embeddings[: self.num_users]
        item_final = final_embeddings[self.num_users :]

        if verbose:
            print(f"  Final user norm: {user_final.norm().item():.3f}")
            print(f"  Final item norm: {item_final.norm().item():.3f}\n")

        if return_layers:
            return user_final, item_final, embeddings_per_layer
        return user_final, item_final

    def forward_score(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        user_emb_all: torch.Tensor,
        item_emb_all: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = user_emb_all[users]
        pos_item_emb = item_emb_all[pos_items]
        neg_item_emb = item_emb_all[neg_items]

        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=-1)

        return pos_scores, neg_scores

    def forward_train(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb_all, item_emb_all = self.get_embeddings(verbose=False)
        return self.forward_score(users, pos_items, neg_items, user_emb_all, item_emb_all)

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        user_emb_all: Optional[torch.Tensor] = None,
        item_emb_all: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if user_emb_all is None or item_emb_all is None:
            return self.forward_train(users, pos_items, neg_items)
        return self.forward_score(users, pos_items, neg_items, user_emb_all, item_emb_all)
