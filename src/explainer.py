# explainer.py

import torch
import numpy as np

class LightGCNExplainer:
    def __init__(self, model, train_dict, movie_names=None, movie_genres=None):
        """
        model: trained LightGCN model
        train_dict: {user_id: set(item_ids)}
        movie_names: {remapped_item_id: title}   <-- must use model's contiguous IDs
        movie_genres: {remapped_item_id: genres}
        """

        self.model = model
        self.train_dict = train_dict
        self.movie_names = movie_names or {}
        self.movie_genres = movie_genres or {}

        self.num_users = model.num_users
        self.num_items = model.num_items
        self.adj = model.adj
        self.n_layers = model.n_layers

        self.model.eval()

        # Precompute embeddings once
        with torch.no_grad():
            self.embs = self.model.propagate()
            self.final_emb = torch.mean(torch.stack(self.embs, dim=0), dim=0)

    # ------------------------------------------------
    # Formatter
    # ------------------------------------------------
    def fmt(self, item_id):
        """Return 'Movie Name (ID) | Genres'"""
        name = self.movie_names.get(item_id, "Unknown")
        genres = self.movie_genres.get(item_id, "Unknown")
        return f"{name} (ID {item_id}) | {genres}"

    # ------------------------------------------------
    # Exact Layer Contribution
    # ------------------------------------------------
    def exact_layer_contributions(self, user_id, item_id):
        """
        Decomposes the LightGCN score into per-layer contributions.

        LightGCN score = dot(mean_k(u_k), mean_l(v_l))
                       = (1/K^2) * sum_k sum_l dot(u_k, v_l)

        We define the contribution of layer k as its marginal share of the
        total score, symmetrically combining its role as both the user-side
        propagation depth and the item-side propagation depth:

            contrib[k] = (1/K^2) * (sum_l dot(u_k, v_l) + sum_l dot(u_l, v_k))
        Returns:
            layer_scores: list of length K (one value per propagation depth)
            contrib_matrix: (K x K) array where [k, l] = dot(u_k, v_l)
        """

        K = len(self.embs)
        item_node = self.num_users + item_id

        contrib_matrix = np.zeros((K, K))
        with torch.no_grad():
            for k in range(K):
                user_k = self.embs[k][user_id]
                for l in range(K):
                    item_l = self.embs[l][item_node]
                    contrib_matrix[k, l] = torch.dot(user_k, item_l).item()

        scale = 1.0 / (K * K)
        layer_scores = []
        for k in range(K):
            # User side: row k summed over all item depths l
            user_side = np.sum(contrib_matrix[k, :])
            # Item side: col k summed over all user depths l (excluding k,k to avoid double-count)
            item_side = np.sum(contrib_matrix[:, k]) - contrib_matrix[k, k]
            layer_scores.append(scale * (user_side + item_side))

        return layer_scores, contrib_matrix

    def dominant_depth(self, user_id, item_id):
        layer_scores, _ = self.exact_layer_contributions(user_id, item_id)
        return int(np.argmax(np.abs(layer_scores))), layer_scores

    # ------------------------------------------------
    # One-hop Attribution
    # ------------------------------------------------
    def one_hop_contributions(self, user_id, item_id, top_k=5):
        """
        Attributes the recommendation score to specific training items in the
        user's history via 1-hop message passing.
        """

        if self.n_layers < 1:
            return []

        if user_id not in self.train_dict:
            return []

        # Use the dominant layer's item embedding for attribution
        k_star, _ = self.dominant_depth(user_id, item_id)
        k_attr = max(1, k_star)  # layer 0 has no propagation; fall back to layer 1

        item_node = self.num_users + item_id
        item_k = self.embs[k_attr][item_node]

        # Compute degree-normalised weight for the user->item_j edge directly
        # from the adjacency, without densifying the entire row.
        user_degree = len(self.train_dict[user_id])
        contributions = []

        with torch.no_grad():
            for item_j in self.train_dict[user_id]:
                node_j = self.num_users + item_j
                item_j_degree = self.adj[node_j].to_dense().count_nonzero().item()
                # Symmetric normalisation weight: 1 / sqrt(deg_u * deg_v)
                weight = 1.0 / np.sqrt(user_degree * max(item_j_degree, 1))
                base_emb = self.embs[0][node_j]
                score = torch.dot(weight * base_emb, item_k).item()
                contributions.append((item_j, score))

        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        return contributions[:top_k]

    # ------------------------------------------------
    # Structured Explanation
    # ------------------------------------------------
    def explain(self, user_id, item_id, top_k=5):
        """
        Returns a structured explanation dict.
        """

        layer_scores, contrib_matrix = self.exact_layer_contributions(user_id, item_id)
        k_star = int(np.argmax(np.abs(layer_scores)))

        explanation = {
            "user_id": user_id,
            "item_id": item_id,
            "item_str": self.fmt(item_id),
            "layer_scores": layer_scores,
            "dominant_depth": k_star,
        }

        if k_star >= 1:
            explanation["top_training_contributors"] = [
                (self.fmt(i), s)
                for i, s in self.one_hop_contributions(user_id, item_id, top_k)
            ]
        else:
            explanation["top_training_contributors"] = None

        return explanation

    # ------------------------------------------------
    # Human Explanation
    # ------------------------------------------------
    def generate_explanation(self, user_id, item_id, top_k=3, _layer_scores=None):
        """
        Generates a human-readable explanation string.
        """

        if _layer_scores is None:
            layer_scores, _ = self.exact_layer_contributions(user_id, item_id)
        else:
            layer_scores = _layer_scores

        abs_scores = [abs(s) for s in layer_scores]

        k_star = int(np.argmax(abs_scores))
        total = sum(abs_scores)
        conf = abs_scores[k_star] / total if total > 0 else 0

        if conf > 0.5:
            strength = "primarily"
        elif conf > 0.35:
            strength = "largely"
        else:
            strength = "partially"

        item_str = self.fmt(item_id)

        # ---------- k=0 ----------
        if k_star == 0:
            return (
                f"{item_str} was recommended {strength} based on your "
                f"overall preference profile learned from past interactions."
            )

        # ---------- k=1 ----------
        elif k_star == 1:

            contribs = self.one_hop_contributions(user_id, item_id, top_k * 3)

            positives = [(i, s) for i, s in contribs if s > 0]
            negatives = [(i, s) for i, s in contribs if s < 0]

            positives.sort(key=lambda x: abs(x[1]), reverse=True)
            negatives.sort(key=lambda x: abs(x[1]), reverse=True)

            pos_items = [self.fmt(i) for i, _ in positives[:top_k]]

            if pos_items:
                text = (
                    f"{item_str} was recommended {strength} because you "
                    f"interacted with {', '.join(pos_items)}, "
                    f"which strongly align with it."
                )
            else:
                text = (
                    f"{item_str} was recommended {strength} due to patterns "
                    f"in your recent interactions."
                )

            if negatives:
                text += (
                    f" Some past interactions (e.g., {self.fmt(negatives[0][0])}) "
                    f"slightly opposed this recommendation."
                )

            return text

        # ---------- k=2 ----------
        elif k_star == 2:
            return (
                f"{item_str} was recommended {strength} because users with "
                f"similar preferences to yours tended to interact with it."
            )

        # ---------- k>=3 ----------
        else:
            return (
                f"{item_str} was recommended {strength} due to broader "
                f"community interaction patterns connected to your activity."
            )
