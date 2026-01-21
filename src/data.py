# data.py


import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import torch

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load MovieLens ratings.
    """
    ratings_df = pd.read_csv(
        path,
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    # drop unused columns
    ratings_df = ratings_df.drop(columns=["rating", "timestamp"])
    return ratings_df


def preprocess_data(
    ratings_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Map user and movie IDs to contiguous indices.
    """
    user_ids = sorted(ratings_df["user_id"].unique())
    movie_ids = sorted(ratings_df["movie_id"].unique())

    user2idx = {u: i for i, u in enumerate(user_ids)}
    idx2user = {i: u for u, i in user2idx.items()}

    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    idx2movie = {i: m for m, i in movie2idx.items()}

    ratings_df["user_id"] = ratings_df["user_id"].map(user2idx)
    ratings_df["movie_id"] = ratings_df["movie_id"].map(movie2idx)

    return ratings_df, user2idx, idx2user, movie2idx, idx2movie


def train_test_split_per_user(
    ratings_df: pd.DataFrame, seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hold out one interaction per user for testing.
    """
    rng = np.random.default_rng(seed)

    train_edges = []
    test_edges = []

    for user_id, user_df in ratings_df.groupby("user_id"):
        items = user_df["movie_id"].tolist()
        test_item = rng.choice(items)
        for item in items:
            if item == test_item:
                test_edges.append((user_id, item))
            else:
                train_edges.append((user_id, item))
    train_interactions = torch.tensor(train_edges, dtype=torch.long)
    test_interactions = torch.tensor(test_edges, dtype=torch.long)

    return train_interactions, test_interactions
