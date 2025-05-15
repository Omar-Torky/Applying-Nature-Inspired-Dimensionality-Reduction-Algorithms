import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from .base import BaseDimReduction

# Set OMP_NUM_THREADS to avoid KMeans memory leak on Windows
os.environ['OMP_NUM_THREADS'] = '1'

class SOMReducer(BaseDimReduction):
    def __init__(self, n_components=2, sigma=1.0, learning_rate=0.5, n_iterations=1000,
                 map_size=(7, 7), n_clusters=3, random_state=None):
        super().__init__(n_components, random_state)
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.map_size = map_size
        self.n_clusters = n_clusters
        self.weights = None
        self.u_matrix = None
        
    def _initialize_weights(self, n_features):
        """Initialize the SOM weight matrix."""
        self.weights = np.random.randn(self.map_size[0], self.map_size[1], n_features)
        
    def _find_bmu(self, x):
        """Find the Best Matching Unit (BMU) for a given input vector."""
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(distances.argmin(), distances.shape)
        
    def _update_weights(self, x, bmu, iteration):
        """Update the weights of the SOM."""
        learning_rate = self.learning_rate * np.exp(-iteration / self.n_iterations)
        sigma = self.sigma * np.exp(-iteration / self.n_iterations)
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                distance = np.linalg.norm(np.array([i, j]) - np.array(bmu))
                influence = np.exp(-(distance ** 2) / (2 * sigma ** 2))
                self.weights[i, j] += learning_rate * influence * (x - self.weights[i, j])
                
    def _compute_u_matrix(self):
        """Compute the U-Matrix."""
        u_matrix = np.zeros(self.map_size)
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                weight = self.weights[i, j]
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.map_size[0] and 0 <= nj < self.map_size[1]:
                        neighbors.append(self.weights[ni, nj])
                if neighbors:
                    distances = [np.linalg.norm(weight - neighbor) for neighbor in neighbors]
                    u_matrix[i, j] = np.mean(distances)
        return u_matrix
        
    def _get_node_labels(self, X, y=None):
        """Get labels for each node in the SOM grid."""
        node_samples = [[[] for _ in range(self.map_size[1])] for _ in range(self.map_size[0])]
        
        for i, x in enumerate(X):
            bmu = self._find_bmu(x)
            if y is not None:
                node_samples[bmu[0]][bmu[1]].append(y[i])
        
        node_labels = np.full(self.map_size, -1)
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if node_samples[i][j] and y is not None:
                    counter = Counter(node_samples[i][j])
                    node_labels[i, j] = counter.most_common(1)[0][0]
                
        return node_labels
        
    def fit_transform(self, X, y=None):
        """Transform X to lower dimensions using SOM."""
        self.set_random_state()
        n_samples, n_features = X.shape
        
        # Apply PCA for initial dimensionality reduction if needed
        if n_features > 5:
            pca = PCA(n_components=5)
            X = pca.fit_transform(X)
            n_features = X.shape[1]
        
        # Initialize weights
        self._initialize_weights(n_features)
        
        # Train the SOM
        for iteration in range(self.n_iterations):
            for i in range(n_samples):
                x = X[i]
                bmu = self._find_bmu(x)
                self._update_weights(x, bmu, iteration)
        
        # Compute U-matrix once
        self.u_matrix = self._compute_u_matrix()
        
        # Transform data points to 2D/3D space
        transformed = np.zeros((n_samples, self.n_components))
        for i in range(n_samples):
            bmu = self._find_bmu(X[i])
            transformed[i, :2] = bmu  # Always use first two components
            if self.n_components > 2:
                transformed[i, 2] = self.u_matrix[bmu]
        
        # Get node labels if y is provided
        if y is not None:
            self.node_labels = self._get_node_labels(X, y)
        
        return self.normalize_output(transformed)
    
    def fit(self, X, y=None):
        """Fit the model (same as fit_transform for this algorithm)."""
        self.fit_transform(X, y)
        return self 