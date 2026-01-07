# data.py

# Load data -> map users and movies into contiguous indices
# test/train split


import pandas as pd

def load_dataset(path : str) -> pd.DataFrame:
    '''
    Load and preprocess the movie ratings data.
    '''
    ratings_df = pd.read_csv(path, sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
    return ratings_df


def preprocess_data(ratings_df : pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocess the ratings data by mapping user and movie IDs to contiguous indices.
    '''
    user_ids = ratings_df['user_id'].unique().tolist()
    movie_ids = ratings_df['movie_id'].unique().tolist()

    user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    ratings_df['user_id'] = ratings_df['user_id'].map(user2idx)
    ratings_df['movie_id'] = ratings_df['movie_id'].map(movie2idx)

    return ratings_df


def save_processed_data(ratings_df : pd.DataFrame, path : str):
    '''
    Save the processed ratings data to a CSV file.
    '''
    ratings_df.to_csv(path, index=False)