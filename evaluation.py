import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import mean_squared_error
from collections import defaultdict

class RecommenderEvaluator:
  
    
    @staticmethod
    def calculate_rmse(y_true: List[float], y_pred: List[float]) -> float:
       
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
     
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    @staticmethod
    def precision_recall_at_k(
        test_ratings: pd.DataFrame,
        recommendations: Dict[int, List[int]],
        k: int = 5,
        threshold: float = 3.5
    ) -> Tuple[float, float]:
       
        precisions = []
        recalls = []
        
        for user_id, rec_items in recommendations.items():
            # Get ground truth relevant items
            user_ratings = test_ratings[test_ratings['user_id'] == user_id]
            relevant_items = set(user_ratings[user_ratings['rating'] >= threshold]['item_id'])
            
            # Get top-k recommended items
            recommended_items = set(rec_items[:k])
            
            # Calculate metrics
            relevant_and_recommended = len(relevant_items & recommended_items)
            
            precision = relevant_and_recommended / k
            recall = relevant_and_recommended / len(relevant_items) if relevant_items else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        return np.mean(precisions), np.mean(recalls)
    
    @staticmethod
    def coverage(
        recommendations: Dict[int, List[int]],
        catalog_size: int
    ) -> float:
        
        all_recommended = set()
        for recs in recommendations.values():
            all_recommended.update(recs)
        return len(all_recommended) / catalog_size
    
    @staticmethod
    def diversity(
        recommendations: Dict[int, List[int]],
        item_similarity: np.ndarray,
        item_index_map: Dict[int, int]
    ) -> float:
      
        total_pairs = 0
        total_dissimilarity = 0.0
        
        for recs in recommendations.values():
            for i in range(len(recs)):
                for j in range(i+1, len(recs)):
                    idx_i = item_index_map[recs[i]]
                    idx_j = item_index_map[recs[j]]
                    similarity = item_similarity[idx_i, idx_j]
                    total_dissimilarity += 1 - similarity
                    total_pairs += 1
                    
        return total_dissimilarity / total_pairs if total_pairs > 0 else 0
    
    @staticmethod
    def evaluate_all(
        test_ratings: pd.DataFrame,
        recommendations: Dict[int, List[int]],
        y_true: List[float],
        y_pred: List[float],
        item_similarity: Optional[np.ndarray] = None,
        item_index_map: Optional[Dict[int, int]] = None,
        catalog_size: Optional[int] = None,
        k: int = 5,
        threshold: float = 3.5
    ) -> Dict[str, float]:
       
        metrics = {}
        
        # Rating prediction metrics
        metrics['rmse'] = RecommenderEvaluator.calculate_rmse(y_true, y_pred)
        metrics['mae'] = RecommenderEvaluator.calculate_mae(y_true, y_pred)
        
        # Ranking metrics
        prec, recall = RecommenderEvaluator.precision_recall_at_k(
            test_ratings, recommendations, k, threshold
        )
        metrics[f'precision@{k}'] = prec
        metrics[f'recall@{k}'] = recall
        
        # Coverage
        if catalog_size:
            metrics['coverage'] = RecommenderEvaluator.coverage(recommendations, catalog_size)
        
        # Diversity
        if item_similarity is not None and item_index_map is not None:
            metrics['diversity'] = RecommenderEvaluator.diversity(
                recommendations, item_similarity, item_index_map
            )
        
        return metrics

# Example usage
if __name__ == '__main__':
    # Mock data for demonstration
    test_ratings = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3],
        'item_id': [101, 102, 101, 103, 102],
        'rating': [4, 3, 5, 2, 4]
    })
    
    recommendations = {
        1: [101, 103, 104],
        2: [102, 103, 105],
        3: [101, 104, 106]
    }
    
    y_true = [4, 3, 5, 2, 4]
    y_pred = [3.8, 3.2, 4.9, 2.1, 3.7]
    
    # Mock similarity data
    item_index_map = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4, 106: 5}
    item_similarity = np.array([
        [1.0, 0.3, 0.2, 0.1, 0.0, 0.0],
        [0.3, 1.0, 0.4, 0.2, 0.1, 0.0],
        [0.2, 0.4, 1.0, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 1.0, 0.4, 0.2],
        [0.0, 0.1, 0.2, 0.4, 1.0, 0.3],
        [0.0, 0.0, 0.1, 0.2, 0.3, 1.0]
    ])
    
    # Evaluate
    metrics = RecommenderEvaluator.evaluate_all(
        test_ratings=test_ratings,
        recommendations=recommendations,
        y_true=y_true,
        y_pred=y_pred,
        item_similarity=item_similarity,
        item_index_map=item_index_map,
        catalog_size=1000,
        k=3
    )
    
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")