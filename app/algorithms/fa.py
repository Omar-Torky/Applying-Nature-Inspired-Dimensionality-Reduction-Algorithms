import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .base import BaseDimReduction

class FAReducer(BaseDimReduction):
    def __init__(self, n_components=2, pop_size=30, max_iter=100, alpha=0.5,
                 beta0=1.0, gamma=1.0, n_clusters=3, random_state=None):
        super().__init__(n_components, random_state)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.n_clusters = n_clusters
        
    def fit_transform(self, X, y=None):
        """Transform X to lower dimensions using Firefly algorithm."""
        self.set_random_state()
        n_samples, n_features = X.shape
        
        # Initialize firefly positions
        fireflies = np.random.randn(self.pop_size, n_features, self.n_components)
        
        def calculate_fitness(solution):
            """Calculate fitness using silhouette score."""
            X_proj = X @ solution
            labels = KMeans(n_clusters=self.n_clusters,
                          random_state=self.random_state).fit_predict(X_proj)
            try:
                return -silhouette_score(X_proj, labels)  # Negative because we minimize
            except:
                return float('inf')
        
        # Calculate initial fitness values
        fitness = np.array([calculate_fitness(f) for f in fireflies])
        best_solution = fireflies[fitness.argmin()].copy()
        best_fitness = fitness.min()
        
        for _ in range(self.max_iter):
            # Update each firefly
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:  # Move firefly i towards j
                        # Calculate distance
                        r = np.linalg.norm(fireflies[i] - fireflies[j])
                        
                        # Calculate attractiveness
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        
                        # Move firefly
                        fireflies[i] += (beta * (fireflies[j] - fireflies[i]) +
                                       self.alpha * (np.random.rand(*fireflies[i].shape) - 0.5))
                        
                        # Update fitness
                        new_fitness = calculate_fitness(fireflies[i])
                        if new_fitness < fitness[i]:
                            fitness[i] = new_fitness
                            
                            # Update best solution
                            if new_fitness < best_fitness:
                                best_solution = fireflies[i].copy()
                                best_fitness = new_fitness
            
            # Reduce alpha (optional)
            self.alpha *= 0.95
        
        # Return the best projection found
        X_reduced = X @ best_solution
        return self.normalize_output(X_reduced)
    
    def fit(self, X, y=None):
        """Fit the model (same as fit_transform for this algorithm)."""
        self.fit_transform(X)
        return self 