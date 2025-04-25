import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix

class CollaborativeFiltering:
 
    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
       
        self.ratings = ratings
        self.movies = movies
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        
    def create_matrices(self, fillna: Optional[float] = None) -> None:
     
        # User-Item matrix (users x movies)
        self.user_item_matrix = self.ratings.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating'
        )
        
        # Item-User matrix (movies x users)
        self.item_user_matrix = self.user_item_matrix.T
        
        if fillna is not None:
            self.user_item_matrix = self.user_item_matrix.fillna(fillna)
            self.item_user_matrix = self.item_user_matrix.fillna(fillna)
    
    def compute_similarities(self, method: str = 'cosine') -> None:
       
        if method not in ['cosine', 'pearson']:
            raise ValueError("Method must be 'cosine' or 'pearson'")
            
        if self.user_item_matrix is None:
            self.create_matrices(fillna=0)
            
        # Convert to sparse matrix for efficiency
        ui_sparse = csr_matrix(self.user_item_matrix.values)
        iu_sparse = csr_matrix(self.item_user_matrix.values)
        
        if method == 'cosine':
            self.user_similarity = cosine_similarity(ui_sparse)
            self.item_similarity = cosine_similarity(iu_sparse)
        else:
            self.user_similarity = np.corrcoef(self.user_item_matrix.fillna(0))
            self.item_similarity = np.corrcoef(self.item_user_matrix.fillna(0))
            
        # Convert to DataFrames with proper indices
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.item_user_matrix.index,
            columns=self.item_user_matrix.index
        )
    
    def user_based_predict(self, user_id: int, item_id: int, k: int = 5) -> float:
       
        if self.user_similarity is None:
            self.compute_similarities()
            
        try:
            # Get similar users
            sim_users = self.user_similarity[user_id].sort_values(ascending=False)[1:k+1]
            
            # Get ratings for the target item
            item_ratings = self.user_item_matrix[item_id]
            
            # Calculate weighted average
            numerator = np.dot(sim_users.values, item_ratings[sim_users.index].values)
            denominator = sim_users.sum()
            
            return numerator / denominator if denominator != 0 else self.user_item_matrix.mean().mean()
            
        except KeyError:
            return self.user_item_matrix.mean().mean()
    
    def item_based_predict(self, user_id: int, item_id: int, k: int = 5) -> float:
        
        if self.item_similarity is None:
            self.compute_similarities()
            
        try:
            # Get similar items
            sim_items = self.item_similarity[item_id].sort_values(ascending=False)[1:k+1]
            
            # Get user's ratings
            user_ratings = self.user_item_matrix.loc[user_id]
            
            # Calculate weighted average
            numerator = np.dot(sim_items.values, user_ratings[sim_items.index].values)
            denominator = sim_items.sum()
            
            return numerator / denominator if denominator != 0 else self.user_item_matrix.mean().mean()
            
        except KeyError:
            return self.user_item_matrix.mean().mean()
    
    def recommend_items(self, user_id: int, n: int = 5, method: str = 'user') -> List[Tuple[str, float]]:
       
        if method not in ['user', 'item']:
            raise ValueError("Method must be 'user' or 'item'")
            
        # Get items not rated by user
        rated_items = set(self.ratings[self.ratings['user_id'] == user_id]['item_id'])
        all_items = set(self.ratings['item_id'])
        unrated_items = all_items - rated_items
        
        # Predict ratings
        predictions = []
        for item in unrated_items:
            if method == 'user':
                pred = self.user_based_predict(user_id, item)
            else:
                pred = self.item_based_predict(user_id, item)
            
            movie_title = self.movies[self.movies['item_id'] == item]['title'].values[0]
            predictions.append((movie_title, pred))
        
        # Return top-N predictions
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    
    def evaluate(self, test_set: pd.DataFrame, method: str = 'user') -> Tuple[float, float]:
        
        y_true = test_set['rating'].values
        y_pred = []
        
        for _, row in test_set.iterrows():
            if method == 'user':
                pred = self.user_based_predict(row['user_id'], row['item_id'])
            else:
                pred = self.item_based_predict(row['user_id'], row['item_id'])
            y_pred.append(pred)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        
        return rmse, mae

# Example usage
if __name__ == '__main__':
    from data_loader import load_and_preprocess
    
    # Load data
    ratings, movies, _ = load_and_preprocess()
    
    # Initialize recommender
    cf = CollaborativeFiltering(ratings, movies)
    cf.create_matrices(fillna=0)
    cf.compute_similarities()
    
    # Generate recommendations
    user_id = 1
    print(f"\nUser-Based Recommendations for user {user_id}:")
    for movie, rating in cf.recommend_items(user_id, method='user'):
        print(f"- {movie}: {rating:.2f}")
    
    print(f"\nItem-Based Recommendations for user {user_id}:")
    for movie, rating in cf.recommend_items(user_id, method='item'):
        print(f"- {movie}: {rating:.2f}")
    
    # Evaluate (example with random test set)
    test_set = ratings.sample(100)
    user_rmse, user_mae = cf.evaluate(test_set, method='user')
    item_rmse, item_mae = cf.evaluate(test_set, method='item')
    
    print(f"\nUser-Based CF - RMSE: {user_rmse:.3f}, MAE: {user_mae:.3f}")
    print(f"Item-Based CF - RMSE: {item_rmse:.3f}, MAE: {item_mae:.3f}")