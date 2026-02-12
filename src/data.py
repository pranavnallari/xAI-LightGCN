# data.py

import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict, List

def load_dataset(path: str) -> pd.DataFrame:
    """Load full MovieLens ratings (keep timestamp for temporal split)."""
    ratings_df = pd.read_csv(path, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'],
                             engine='python', encoding='latin-1')
    return ratings_df

def preprocess_data(ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int], ...]:
    """Map IDs to contiguous indices."""
    user_ids = sorted(ratings_df['user_id'].unique())
    movie_ids = sorted(ratings_df['movie_id'].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    movie2idx = {m: i for i, m in enumerate(movie_ids)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2movie = {i: m for m, i in movie2idx.items()}
    ratings_df['user_id'] = ratings_df['user_id'].map(user2idx)
    ratings_df['movie_id'] = ratings_df['movie_id'].map(movie2idx)
    ratings_df = ratings_df.drop(columns=['rating'])  # Implicit
    return ratings_df, user2idx, idx2user, movie2idx, idx2movie

def temporal_split(ratings_df, val_ratio=0.1, test_ratio=0.1):
    ratings_df = ratings_df.sort_values('timestamp')

    train_edges, val_edges, test_edges = [], [], []

    for user_id, user_df in ratings_df.groupby('user_id'):
        items = user_df['movie_id'].tolist()

        if len(items) < 3:
            train_edges.extend([(user_id, i) for i in items])
            continue

        n_test = max(1, int(len(items) * test_ratio))
        n_val = max(1, int(len(items) * val_ratio))

        test_items = items[-n_test:]
        val_items = items[-(n_test + n_val):-n_test]
        train_items = items[:-(n_test + n_val)]

        train_edges.extend([(user_id, i) for i in train_items])
        val_edges.extend([(user_id, i) for i in val_items])
        test_edges.extend([(user_id, i) for i in test_items])

    return (
        torch.tensor(train_edges, dtype=torch.long),
        torch.tensor(val_edges, dtype=torch.long),
        torch.tensor(test_edges, dtype=torch.long),
    )
