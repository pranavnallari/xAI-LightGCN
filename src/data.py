# data.py

import pandas as pd
import numpy as np
import random


def load_movielens_1m(path, seed=42):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep="::", names=names, engine="python")
    df = df[['user_id', 'item_id']]
    user_map = {u: i for i, u in enumerate(df['user_id'].unique())}
    item_map = {i: j for j, i in enumerate(df['item_id'].unique())}
    df['user_id'] = df['user_id'].map(user_map)
    df['item_id'] = df['item_id'].map(item_map)
    num_users = len(user_map)
    num_items = len(item_map)
    df = df.sample(frac=1, random_state=seed)
    n_total = len(df)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    train_data = list(zip(train_df['user_id'], train_df['item_id']))
    val_data = list(zip(val_df['user_id'], val_df['item_id']))
    test_data = list(zip(test_df['user_id'], test_df['item_id']))
    return train_data, val_data, test_data, num_users, num_items, item_map


def load_movie_metadata(movies_file):
    """
    Returns:
        movie_names: {raw_movie_id: "Title"}
        movie_genres: {raw_movie_id: "Genre1|Genre2"}

    NOTE: IDs here are raw MovieLens IDs. After calling load_movielens_1m(),
    use the returned item_map to remap these to contiguous model IDs:

        remapped_names  = {item_map[k]: v for k, v in movie_names.items()  if k in item_map}
        remapped_genres = {item_map[k]: v for k, v in movie_genres.items() if k in item_map}
    """
    movie_names = {}
    movie_genres = {}
    with open(movies_file, encoding="latin-1") as f:
        for line in f:
            movie_id, title, genres = line.strip().split("::")
            movie_id = int(movie_id)
            movie_names[movie_id] = title
            movie_genres[movie_id] = genres
    return movie_names, movie_genres
