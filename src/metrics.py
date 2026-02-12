# metrics.py

import numpy as np


def compute_metrics(ranked_list, true_items, k):

    # Hit@K
    hit = int(any(i in ranked_list for i in true_items))

    # Precision@K
    precision = sum(i in true_items for i in ranked_list) / k

    # NDCG@K
    dcg = 0.0
    for rank, item in enumerate(ranked_list):
        if item in true_items:
            dcg += 1.0 / np.log2(rank + 2)

    ideal_dcg = sum(
        1.0 / np.log2(i + 2)
        for i in range(min(len(true_items), k))
    )

    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    # MRR@K
    rr = 0.0
    for rank, item in enumerate(ranked_list):
        if item in true_items:
            rr = 1.0 / (rank + 1)
            break

    return hit, precision, ndcg, rr
