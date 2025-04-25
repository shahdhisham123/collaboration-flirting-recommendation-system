import pandas as pd
import numpy as np
from typing import Tuple, Dict

def load_raw_data(data_dir: str = 'C:/Users/Zenab/Desktop') -> Dict[str, pd.DataFrame]:
    
    try:
        # Load ratings data
        ratings = pd.read_csv(
            f'{data_dir}u.data',
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
        
        # Load movie metadata
        movies = pd.read_csv(
            f'{data_dir}u.item',
            sep='|',
            encoding='latin-1',
            usecols=[0, 1, 2],
            names=['item_id', 'title', 'release_date']
        )
        
        # Load user demographics
        users = pd.read_csv(
            f'{data_dir}u.user',
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
        
        return {
            'ratings': ratings,
            'movies': movies,
            'users': users
        }
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset files not found in {data_dir}. Please download the MovieLens 100K dataset") from e

def preprocess_data(data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    # Process ratings
    ratings = data['ratings'].copy()
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    
    # Process movies
    movies = data['movies'].copy()
    movies['release_year'] = movies['release_date'].str.extract(r'(\d{4})').astype(float)
    movies.drop('release_date', axis=1, inplace=True)
    
    # Process users
    users = data['users'].copy()
    users['age_group'] = pd.cut(
        users['age'],
        bins=[0, 18, 25, 35, 50, 100],
        labels=['<18', '18-25', '26-35', '36-50', '50+']
    )
    
    return ratings, movies, users

def create_user_item_matrix(ratings: pd.DataFrame, 
                          fillna: float = None) -> pd.DataFrame:
   
    matrix = ratings.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating'
    )
    
    if fillna is not None:
        matrix = matrix.fillna(fillna)
        
    return matrix

def load_and_preprocess(data_dir: str = 'data/') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    raw_data = load_raw_data(data_dir)
    return preprocess_data(raw_data)

# Example usage
if __name__ == '__main__':
    try:
        ratings, movies, users = load_and_preprocess()
        
        print("Ratings data sample:")
        print(ratings.head())
        
        print("\nMovies data sample:")
        print(movies.head())
        
        print("\nUsers data sample:")
        print(users.head())
        
        user_item_matrix = create_user_item_matrix(ratings, fillna=ratings['rating'].mean())
        print("\nUser-item matrix shape:", user_item_matrix.shape)
        
    except Exception as e:
        print(f"Error loading data: {e}")