import numpy as np
import os
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .base import BaseDimReduction

# Set OMP_NUM_THREADS to avoid KMeans memory leak on Windows
os.environ['OMP_NUM_THREADS'] = '1'

class BATReducer(BaseDimReduction):
    def __init__(self, n_components=2, n_bats=40, max_iter=100, f_min=0.0, f_max=2.0,
                 alpha=0.9, gamma=0.9, A0=1.0, r0=0.5, n_neighbors=10, n_clusters=3,
                 random_state=None):
        super().__init__(n_components, random_state)
        self.n_bats = n_bats
        self.max_iter = max_iter
        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha
        self.gamma = gamma
        self.A0 = A0
        self.r0 = r0
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        
    def _project_data(self, position):
        """Project data using the bat's position."""
        return self.X @ position.reshape(self.n_features, self.n_components)
        
    def _calculate_fitness(self, position):
        """Calculate fitness using trustworthiness and clustering quality."""
        X_proj = self._project_data(position)
        
        # Calculate trustworthiness
        trust = trustworthiness(self.X, X_proj, n_neighbors=self.n_neighbors)
        
        try:
            # Determine optimal number of clusters
            max_clusters = min(self.n_clusters, len(np.unique(X_proj, axis=0)))
            if max_clusters < 2:
                return 1.0  # Maximum dissimilarity
                
            # Try clustering with different numbers of clusters
            best_sil = float('-inf')
            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, 
                              random_state=self.random_state,
                              n_init=1)
                labels = kmeans.fit_predict(X_proj)
                
                # Only calculate silhouette if we have enough unique points
                if len(np.unique(labels)) > 1:
                    sil = silhouette_score(X_proj, labels)
                    best_sil = max(best_sil, sil)
            
            if best_sil > float('-inf'):
                # Combine trustworthiness and silhouette score
                return 1.0 - (0.5 * trust + 0.5 * best_sil)
            return 1.0  # Maximum dissimilarity
        except:
            return 1.0  # Maximum dissimilarity
        
    def fit_transform(self, X, y=None):
        """Transform X to lower dimensions using BAT algorithm."""
        self.set_random_state()
        self.X = X
        self.n_samples, self.n_features = X.shape
        dim = self.n_features * self.n_components
        
        # Initialize bat positions and velocities
        pos = np.random.uniform(-1, 1, (self.n_bats, dim))
        vel = np.zeros_like(pos)
        
        # Initialize loudness and pulse rate
        loud = np.full(self.n_bats, self.A0)
        pulse = np.full(self.n_bats, self.r0)
        
        # Calculate initial fitness
        fit = np.array([self._calculate_fitness(p) for p in pos])
        best = pos[fit.argmin()].copy()
        best_fit = fit.min()
        
        for _ in range(self.max_iter):
            # Generate new solutions
            freq = self.f_min + (self.f_max - self.f_min) * np.random.rand(self.n_bats)
            vel = vel + (pos - best) * freq[:, None]
            new_pos = pos + vel
            
            # Local search
            pulse_mask = np.random.rand(*pos.shape) < pulse[:, None]
            new_pos = np.where(pulse_mask, new_pos, pos)
            
            # Evaluate new solutions
            new_fit = np.array([self._calculate_fitness(p) for p in new_pos])
            
            # Update if better and accepted
            accept = (new_fit < fit) & (np.random.rand(self.n_bats) < loud)
            pos[accept] = new_pos[accept]
            fit[accept] = new_fit[accept]
            
            # Update loudness and pulse rate
            loud *= self.alpha
            pulse = self.r0 * (1 - np.exp(-self.gamma * np.arange(1, self.n_bats + 1)))
            
            # Update best solution
            best_idx = fit.argmin()
            if fit[best_idx] < best_fit:
                best = pos[best_idx].copy()
                best_fit = fit[best_idx]
        
        # Return the best projection found
        X_reduced = self._project_data(best)
        return self.normalize_output(X_reduced)
    
    def fit(self, X, y=None):
        """Fit the model (same as fit_transform for this algorithm)."""
        self.fit_transform(X)
        return self 