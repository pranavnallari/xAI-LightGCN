# data.py

import pandas as pd
import numpy as np
import random


def load_movielens_1m(path, seed=42):

    names = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep="::", names=names, engine="python")

    df = df[['user_id', 'item_id']]

    # Remap IDs
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

    return train_data, val_data, test_data, num_users, num_items
