from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class BaseDimReduction(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        
    def set_random_state(self):
        """Set random state for reproducibility."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
    def normalize_output(self, X):
        """Normalize output to [0, 1] range."""
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        return X_norm 