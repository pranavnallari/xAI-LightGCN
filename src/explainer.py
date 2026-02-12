# explainer.py

import torch
import numpy as np


class LightGCNExplainer:
    def __init__(self, model, train_dict):
        """
        model: trained LightGCN model (frozen)
        train_dict: {user_id: set(training_item_ids)}
        """

        self.model = model
        self.train_dict = train_dict
        self.num_users = model.num_users
        self.num_items = model.num_items
        self.adj = model.adj
        self.n_layers = model.n_layers

        self.model.eval()

        # Precompute layer-wise embeddings once
        with torch.no_grad():
            self.embs = self.model.propagate()
            self.final_emb = torch.mean(
                torch.stack(self.embs, dim=0), dim=0
            )

    # ------------------------------------------------
    # 1️⃣ Depth Contribution Analysis
    # ------------------------------------------------
    def layerwise_score(self, user_id, item_id):
        """
        Returns:
            scores_per_layer: list of contributions per depth
        """

        scores = []

        for k in range(len(self.embs)):

            user_k = self.embs[k][user_id]
            item_k = self.embs[k][self.num_users + item_id]

            score_k = torch.dot(user_k, item_k).item()
            scores.append(score_k)

        return scores

    def dominant_depth(self, user_id, item_id):
        scores = self.layerwise_score(user_id, item_id)
        k_star = int(np.argmax(np.abs(scores)))
        return k_star, scores

    # ------------------------------------------------
    # 2️⃣ 1-hop Attribution (k = 1 case)
    # ------------------------------------------------
    def one_hop_contributions(self, user_id, item_id, top_k=5):
        """
        Computes contribution of each training item of the user
        to the recommendation score at k=1 layer.
        """

        if user_id not in self.train_dict:
            return []

        contributions = []

        # 1-hop embedding for item
        item_k = self.embs[1][self.num_users + item_id]

        # Get normalization weights from adjacency
        user_row = self.adj[user_id].to_dense()

        training_items = self.train_dict[user_id]

        for item_j in training_items:

            node_j = self.num_users + item_j

            weight = user_row[node_j].item()

            base_emb = self.embs[0][node_j]

            contribution_vec = weight * base_emb

            contribution_score = torch.dot(
                contribution_vec,
                item_k
            ).item()

            contributions.append((item_j, contribution_score))

        # Sort by absolute contribution
        contributions.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return contributions[:top_k]

    # ------------------------------------------------
    # 3️⃣ Full Explanation Summary
    # ------------------------------------------------
    def explain(self, user_id, item_id, top_k=5):

        k_star, layer_scores = self.dominant_depth(user_id, item_id)

        explanation = {
            "user_id": user_id,
            "item_id": item_id,
            "layer_scores": layer_scores,
            "dominant_depth": k_star,
        }

        if k_star == 1:
            explanation["top_training_contributors"] = \
                self.one_hop_contributions(user_id, item_id, top_k)

        else:
            explanation["top_training_contributors"] = None

        return explanation
