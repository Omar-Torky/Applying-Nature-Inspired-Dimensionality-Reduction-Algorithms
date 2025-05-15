import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .base import BaseDimReduction

# Set OMP_NUM_THREADS to avoid KMeans memory leak on Windows
os.environ['OMP_NUM_THREADS'] = '1'

class PSOReducer(BaseDimReduction):
    def __init__(self, n_components=2, pop_size=90, max_iter=90, w=0.8, c1=1.2, c2=1.2,
                 n_clusters=3, k_trust=10, alpha=0.5, random_state=None):
        super().__init__(n_components, random_state)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_clusters = n_clusters
        self.k_trust = k_trust
        self.alpha = alpha
        
    def _calculate_trustworthiness(self, X_high, X_low):
        """Calculate trustworthiness score."""
        n = X_high.shape[0]
        hi = NearestNeighbors(n_neighbors=n-1).fit(X_high).kneighbors(return_distance=False)
        lo = NearestNeighbors(n_neighbors=self.k_trust).fit(X_low).kneighbors(return_distance=False)
        t = 0
        for i in range(n):
            ranks = np.array([np.where(hi[i] == j)[0][0] for j in lo[i]])
            t += np.sum((ranks > self.k_trust-1) * (ranks - (self.k_trust-1)))
        return 1 - (2 / (n * self.k_trust * (2*n - 3*self.k_trust - 1))) * t
        
    def _calculate_fitness(self, X_proj):
        """Calculate combined fitness using silhouette score and trustworthiness."""
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
        """Transform X to lower dimensions using PSO."""
        self.set_random_state()
        self.X = X  # Store for trustworthiness calculation
        n_samples, n_features = X.shape
        
        # Initialize particles
        particles = np.random.randn(self.pop_size, n_features, self.n_components)
        velocities = np.zeros_like(particles)
        pbest = particles.copy()
        pbest_fitness = np.full(self.pop_size, float('inf'))
        gbest = particles[0].copy()
        gbest_fitness = float('inf')
        
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # Project data
                X_proj = X @ particles[i]
                
                # Calculate fitness
                fitness = self._calculate_fitness(X_proj)
                
                # Update personal best
                if fitness < pbest_fitness[i]:
                    pbest_fitness[i] = fitness
                    pbest[i] = particles[i].copy()
                
                # Update global best
                if fitness < gbest_fitness:
                    gbest_fitness = fitness
                    gbest = particles[i].copy()
            
            # Update velocities and positions
            r1, r2 = np.random.rand(), np.random.rand()
            velocities = (self.w * velocities + 
                        self.c1 * r1 * (pbest - particles) +
                        self.c2 * r2 * (gbest - particles))
            particles += velocities
        
        # Return the best projection found
        X_reduced = X @ gbest
        return self.normalize_output(X_reduced)
    
    def fit(self, X, y=None):
        """Fit the model (same as fit_transform for this algorithm)."""
        self.fit_transform(X)
        return self 