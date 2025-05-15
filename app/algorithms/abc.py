import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from .base import BaseDimReduction

# Set OMP_NUM_THREADS to avoid KMeans memory leak on Windows
os.environ['OMP_NUM_THREADS'] = '1'

class ABCReducer(BaseDimReduction):
    def __init__(self, n_components=2, pop_size=30, max_iter=100, limit=10,
                 n_clusters=3, k_trust=10, alpha=0.5, random_state=None):
        super().__init__(n_components, random_state)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit = limit
        self.n_clusters = n_clusters
        self.k_trust = k_trust
        self.alpha = alpha
        
    def _calculate_trustworthiness(self, X_high, X_low):
        """Calculate trustworthiness score."""
        return trustworthiness(X_high, X_low, n_neighbors=self.k_trust)
        
    def _calculate_fitness(self, solution):
        """Calculate combined fitness using silhouette score and trustworthiness."""
        X_proj = self.X @ solution
        try:
            # Determine optimal number of clusters
            max_clusters = min(self.n_clusters, len(np.unique(X_proj, axis=0)))
            if max_clusters < 2:
                return float('inf')
                
            # Try clustering with different numbers of clusters
            best_score = float('-inf')
            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters,
                              random_state=self.random_state,
                              n_init=1)
                labels = kmeans.fit_predict(X_proj)
                
                # Only calculate silhouette if we have enough unique points
                if len(np.unique(labels)) > 1:
                    sil = silhouette_score(X_proj, labels)
                    trust = self._calculate_trustworthiness(self.X, X_proj)
                    score = self.alpha * sil + (1 - self.alpha) * trust
                    best_score = max(best_score, score)
            
            return -best_score if best_score > float('-inf') else float('inf')
        except:
            return float('inf')
        
    def fit_transform(self, X, y=None):
        """Transform X to lower dimensions using ABC algorithm."""
        self.set_random_state()
        self.X = X
        n_samples, n_features = X.shape
        
        # Initialize food sources (solutions)
        food = np.random.randn(self.pop_size, n_features, self.n_components)
        trial = np.zeros(self.pop_size)
        
        # Calculate initial fitness values
        fitness = np.array([self._calculate_fitness(f) for f in food])
        best_solution = food[fitness.argmin()].copy()
        best_fitness = fitness.min()
        
        for _ in range(self.max_iter):
            # Employed Bees Phase
            for i in range(self.pop_size):
                # Generate new solution
                k = np.random.choice([j for j in range(self.pop_size) if j != i])
                phi = np.random.uniform(-1, 1, food[i].shape)
                v = food[i] + phi * (food[i] - food[k])
                
                # Calculate fitness and update if better
                v_fit = self._calculate_fitness(v)
                if v_fit < fitness[i]:
                    food[i] = v
                    fitness[i] = v_fit
                    trial[i] = 0
                else:
                    trial[i] += 1
            
            # Onlooker Bees Phase
            prob = (1 / (1 + fitness)) / np.sum(1 / (1 + fitness))
            for i in range(self.pop_size):
                if np.random.random() < prob[i]:
                    k = np.random.choice([j for j in range(self.pop_size) if j != i])
                    phi = np.random.uniform(-1, 1, food[i].shape)
                    v = food[i] + phi * (food[i] - food[k])
                    
                    v_fit = self._calculate_fitness(v)
                    if v_fit < fitness[i]:
                        food[i] = v
                        fitness[i] = v_fit
                        trial[i] = 0
                    else:
                        trial[i] += 1
            
            # Scout Bees Phase
            abandoned = trial > self.limit
            if np.any(abandoned):
                food[abandoned] = np.random.randn(abandoned.sum(), n_features, self.n_components)
                fitness[abandoned] = np.array([self._calculate_fitness(f) for f in food[abandoned]])
                trial[abandoned] = 0
            
            # Update best solution
            min_idx = fitness.argmin()
            if fitness[min_idx] < best_fitness:
                best_solution = food[min_idx].copy()
                best_fitness = fitness[min_idx]
        
        # Return the best projection found
        X_reduced = X @ best_solution
        return self.normalize_output(X_reduced)
    
    def fit(self, X, y=None):
        """Fit the model (same as fit_transform for this algorithm)."""
        self.fit_transform(X)
        return self 