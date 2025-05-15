import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler
from .base import BaseDimReduction
import matplotlib.pyplot as plt

# Set OMP_NUM_THREADS to avoid KMeans memory leak on Windows
os.environ['OMP_NUM_THREADS'] = '1'

class ACO(BaseDimReduction):
    def __init__(self, n_components=2, n_ants=30, n_iterations=100, evaporation_rate=0.1,
                 k_attract=30, k_repulse=50, attraction_strength=0.5, repulsion_strength=0.1,
                 alpha=0.5, random_state=None):
        super().__init__(n_components, random_state)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.k_attract = k_attract
        self.k_repulse = k_repulse
        self.attraction_strength = attraction_strength
        self.repulsion_strength = repulsion_strength
        self.alpha = alpha
        self.centering_interval = 20

    def _calculate_forces(self, Y, indices_attract, indices_repulse, i):
        """Calculate attraction and repulsion forces for a point."""
        neighbors_attract = indices_attract[i][1:]  # Exclude self
        neighbors_repulse = indices_repulse[i][1:]  # Exclude self
        
        # Calculate attraction forces
        attraction = np.zeros(self.n_components)
        for j in neighbors_attract:
            diff = Y[j] - Y[i]
            norm = np.linalg.norm(diff) + 1e-6
            attraction += (diff / norm) * self.attraction_strength
            
        # Calculate repulsion forces
        repulsion = np.zeros(self.n_components)
        for j in neighbors_repulse:
            if j != i:
                diff = Y[i] - Y[j]
                norm = np.linalg.norm(diff) + 1e-6
                repulsion += (diff / (norm ** 2)) * self.repulsion_strength
                
        return attraction + repulsion

    def _create_ant_solution(self, Y, pheromone, indices_attract, indices_repulse, move_scale):
        """Create a solution for one ant."""
        ant_Y = Y.copy()
        n_samples = Y.shape[0]
        
        for i in range(n_samples):
            # Calculate combined forces
            forces = self._calculate_forces(ant_Y, indices_attract, indices_repulse, i)
            
            # Update position with forces, pheromone influence, and random noise
            ant_Y[i] += (move_scale * forces + 
                        self.alpha * pheromone[i] + 
                        np.random.randn(self.n_components) * 0.01)
            
        return ant_Y

    def _calculate_cost(self, Y):
        """Calculate the cost (variance) of a solution."""
        return np.var(Y)

    def fit_transform(self, X, y=None):
        """Transform X to lower dimensions using ACO with attraction-repulsion."""
        self.set_random_state()
        X = StandardScaler().fit_transform(X)
        n_samples = X.shape[0]
        
        # Initialize neighbor calculations
        nbrs_attract = NearestNeighbors(n_neighbors=self.k_attract + 1).fit(X)
        indices_attract = nbrs_attract.kneighbors(X, return_distance=False)
        
        nbrs_repulse = NearestNeighbors(n_neighbors=self.k_repulse + 1).fit(X)
        indices_repulse = nbrs_repulse.kneighbors(X, return_distance=False)
        
        # Initialize low-dimensional positions and pheromone
        Y = np.random.randn(n_samples, self.n_components)
        pheromone = np.ones((n_samples, self.n_components))
        
        best_Y = Y.copy()
        best_cost = self._calculate_cost(Y)
        
        for iteration in range(self.n_iterations):
            # Scale movement based on iteration progress
            move_scale = 0.1 * (1 - iteration / self.n_iterations)
            
            # Generate solutions for each ant
            new_positions = []
            costs = []
            
            for _ in range(self.n_ants):
                ant_Y = self._create_ant_solution(Y, pheromone, indices_attract, indices_repulse, move_scale)
                cost = self._calculate_cost(ant_Y)
                new_positions.append(ant_Y)
                costs.append(cost)
            
            # Update best solution
            best_idx = np.argmax(costs)
            if costs[best_idx] > best_cost:
                best_cost = costs[best_idx]
                best_Y = new_positions[best_idx].copy()
            
            # Update current position and pheromone
            Y = new_positions[best_idx].copy()
            pheromone = ((1 - self.evaporation_rate) * pheromone + 
                        (new_positions[best_idx] - Y) * 0.1)
            
            # Periodic centering and scaling
            if (iteration + 1) % self.centering_interval == 0:
                Y -= Y.mean(axis=0)
                scale = np.linalg.norm(Y, axis=1).max()
                if scale > 0:
                    Y = Y / scale * 10
        
        # Final centering and scaling
        Y = (Y - Y.mean(axis=0)) * 1.5
        
        if y is not None:
            sil_score = silhouette_score(Y, y)
            trust = trustworthiness(X, Y, n_neighbors=5)
            print(f"Silhouette Score: {sil_score:.4f}")
            print(f"Trustworthiness: {trust:.4f}")
            
            # Visualization
            plt.figure(figsize=(8, 6))
            for label in np.unique(y):
                plt.scatter(Y[y == label, 0], Y[y == label, 1], label=f'Class {label}')
            plt.legend()
            plt.xlabel('Component 1')
            plt.ylabel('Component 2')
            plt.title('ACO Dimension Reduction')
            plt.tight_layout()
            plt.show()
        
        return Y
    
    def fit(self, X, y=None):
        """Fit the model (same as fit_transform for this algorithm)."""
        self.fit_transform(X, y)
        return self 