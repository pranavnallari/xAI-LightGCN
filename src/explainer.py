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
    # Exact Layer Contribution (Full Decomposition)
    # ------------------------------------------------
    def exact_layer_contributions(self, user_id, item_id):

        K = len(self.embs)
        contrib_matrix = np.zeros((K, K))

        for k in range(K):
            user_k = self.embs[k][user_id]

            for l in range(K):
                item_l = self.embs[l][self.num_users + item_id]

                score = torch.dot(user_k, item_l).item()
                contrib_matrix[k, l] = score

        # Total contribution per layer
        layer_scores = []

        for k in range(K):

            row_sum = np.sum(contrib_matrix[k, :])
            col_sum = np.sum(contrib_matrix[:, k])

            # subtract diagonal once (counted twice)
            layer_score = row_sum + col_sum - contrib_matrix[k, k]

            layer_scores.append(layer_score)

        return layer_scores, contrib_matrix


    def dominant_depth(self, user_id, item_id):

        layer_scores, _ = self.exact_layer_contributions(user_id, item_id)

        k_star = int(np.argmax(np.abs(layer_scores)))

        return k_star, layer_scores

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

    # ------------------------------------------------
    # Concise Human-Readable Explanation
    # ------------------------------------------------
    def generate_explanation(self, user_id, item_id, top_k=3):

        layer_scores, _ = self.exact_layer_contributions(user_id, item_id)
        abs_scores = [abs(s) for s in layer_scores]

        k_star = int(np.argmax(abs_scores))

        total_magnitude = sum(abs_scores)
        confidence = abs_scores[k_star] / total_magnitude if total_magnitude > 0 else 0

        # Confidence wording
        if confidence > 0.5:
            strength_word = "primarily"
        elif confidence > 0.35:
            strength_word = "largely"
        else:
            strength_word = "partially"

        # ------------------------------------------------
        # k* = 0 → Base Embedding
        # ------------------------------------------------
        if k_star == 0:
            return (
                f"Item {item_id} was recommended {strength_word} "
                f"based on your overall preference profile learned from past interactions."
            )

        # ------------------------------------------------
        # k* = 1 → Direct Neighbor Attribution
        # ------------------------------------------------
        elif k_star == 1:

            # Compute ALL one-hop contributions (no slicing)
            if user_id not in self.train_dict:
                return (
                    f"Item {item_id} was recommended due to interaction patterns "
                    f"in your activity history."
                )

            contributions = []

            item_k = self.embs[1][self.num_users + item_id]
            user_row = self.adj[user_id].to_dense()

            for item_j in self.train_dict[user_id]:
                node_j = self.num_users + item_j
                weight = user_row[node_j].item()
                base_emb = self.embs[0][node_j]

                contribution_vec = weight * base_emb
                contribution_score = torch.dot(
                    contribution_vec, item_k
                ).item()

                contributions.append((item_j, contribution_score))

            # Separate positives and negatives
            positives = [(i, s) for i, s in contributions if s > 0]
            negatives = [(i, s) for i, s in contributions if s < 0]

            # Sort positives by magnitude
            positives.sort(key=lambda x: abs(x[1]), reverse=True)
            negatives.sort(key=lambda x: abs(x[1]), reverse=True)

            positive_items = [str(i) for i, _ in positives[:top_k]]

            if positive_items:
                explanation = (
                    f"Item {item_id} was recommended {strength_word} because you "
                    f"interacted with items {', '.join(positive_items)}, "
                    f"which strongly align with it."
                )
            else:
                explanation = (
                    f"Item {item_id} was recommended {strength_word} due to "
                    f"patterns in your recent interactions."
                )

            if negatives:
                explanation += (
                    f" Some past interactions (e.g., item {negatives[0][0]}) "
                    f"slightly opposed this recommendation."
                )

            return explanation

        # ------------------------------------------------
        # k* = 2 → Collaborative Influence
        # ------------------------------------------------
        elif k_star == 2:
            return (
                f"Item {item_id} was recommended {strength_word} because users "
                f"with similar preferences to yours tended to interact with it."
            )

        # ------------------------------------------------
        # k* ≥ 3 → Community Diffusion
        # ------------------------------------------------
        else:
            return (
                f"Item {item_id} was recommended {strength_word} due to broader "
                f"community interaction patterns connected to your activity."
            )
