import torch

class LightGCNExplainer:
    def __init__(self, model, user_pos_items):
        """
        model: trained LightGCN
        user_pos_items: dict {user_id: set(item_ids)} for graph traversal
        """
        self.model = model
        self.user_pos_items = user_pos_items
        self.num_users = model.num_users

    @torch.no_grad()
    def explain(self, user_id, item_id, max_hops=None):
        """
        Returns:
          - k_star (dominant layer)
          - layer_contributions
          - example propagation paths (list)
        """

        self.model.eval()

        # ---- Step 1: get embeddings + layers
        user_emb, item_emb, layer_embs = self.model.get_embeddings(
            return_layers=True
        )

        K = len(layer_embs) - 1

        # ---- Step 2: extract per-layer embeddings
        u_layers = [
            layer_embs[k][: self.num_users][user_id]
            for k in range(K + 1)
        ]
        i_layers = [
            layer_embs[k][self.num_users :][item_id]
            for k in range(K + 1)
        ]

        # ---- Step 3: layer-wise scores
        scores = torch.tensor([
            torch.dot(u_layers[k], i_layers[k]).item()
            for k in range(K + 1)
        ])
        scores = torch.clamp(scores, min=0)

        if scores.sum() == 0:
            return {
                "k_star": None,
                "contributions": None,
                "paths": []
            }

        contributions = scores / scores.sum()
        k_star = int(torch.argmax(contributions).item())

        if max_hops is not None:
            k_star = min(k_star, max_hops)

        # ---- Step 4: graph-based path extraction
        paths = self._extract_paths(user_id, item_id, k_star)

        return {
            "k_star": k_star,
            "contributions": contributions.tolist(),
            "paths": paths
        }

    # --------------------------------------------------

    def _extract_paths(self, user_id, target_item, hops):
        """
        BFS constrained to `hops`, bipartite-aware.
        Returns small list of paths.
        """

        if hops == 0:
            return []

        paths = []
        queue = [(user_id, 0, [("U", user_id)])]

        while queue:
            node, depth, path = queue.pop(0)

            if depth == hops:
                if path[-1] == ("I", target_item):
                    paths.append(path)
                continue

            if depth % 2 == 0:
                # User → Item
                for item in self.user_pos_items.get(node, []):
                    queue.append(
                        (item, depth + 1, path + [("I", item)])
                    )
            else:
                # Item → User
                for u, items in self.user_pos_items.items():
                    if node in items:
                        queue.append(
                            (u, depth + 1, path + [("U", u)])
                        )

            if len(paths) >= 3:   # cap explanations
                break

        return paths
