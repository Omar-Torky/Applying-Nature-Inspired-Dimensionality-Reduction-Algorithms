from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.cluster import KMeans
import time

def calculate_metrics(X_original, X_reduced, n_clusters, random_state):
    """Calculate evaluation metrics for dimensionality reduction."""
    start_time = time.time()
    
    # Clustering metrics
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X_reduced)
    silhouette = silhouette_score(X_reduced, kmeans.labels_)
    
    # Trustworthiness
    trust = trustworthiness(X_original, X_reduced, n_neighbors=5)
    
    # Computation time
    computation_time = time.time() - start_time
    
    return {
        'silhouette': silhouette,
        'trustworthiness': trust,
        'time': computation_time
    } 