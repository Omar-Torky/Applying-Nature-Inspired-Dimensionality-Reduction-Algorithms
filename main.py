import streamlit as st
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Import our modules
from app.config import setup_page_config, apply_custom_css
from app.data_loader import load_data, get_dataset_info
from app.components.sidebar import create_sidebar
from app.components.tabs import create_visualization_tab, create_metrics_tab
from app.algorithms.aco import ACO
from app.algorithms.som import SOMReducer
from app.algorithms.pso import PSOReducer
from app.algorithms.bat import BATReducer
from app.algorithms.abc import ABCReducer
from app.algorithms.fa import FAReducer
from app.utils.metrics import calculate_metrics

# Must be the first Streamlit command
setup_page_config()
apply_custom_css()

# Title
st.markdown("<h1 class='main-header'>Nature-Inspired Dimensionality Reduction for Data Visualization</h1>", unsafe_allow_html=True)

# Load data
X, y, feature_names, target_names = load_data()

# Create sidebar and get parameters
algorithms, params, viz_params = create_sidebar()

# Set random seeds
random_state = params['random_state']
np.random.seed(random_state)
random.seed(random_state)

# Initialize data and metrics dictionaries
reduced_data = {}
metrics = {}

# Run selected algorithms
if algorithms['PCA']:
    pca = PCA(n_components=params['n_components'], random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    reduced_data['PCA'] = {
        'embedding': X_pca,
        'labels': y,
        'target_names': target_names
    }
    
    metrics['PCA'] = calculate_metrics(X, X_pca, params['n_clusters'], random_state)
    metrics['PCA']['variance_explained'] = sum(pca.explained_variance_ratio_)

if algorithms['t-SNE']:
    tsne = TSNE(n_components=params['n_components'], random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    
    reduced_data['t-SNE'] = {
        'embedding': X_tsne,
        'labels': y,
        'target_names': target_names
    }
    
    metrics['t-SNE'] = calculate_metrics(X, X_tsne, params['n_clusters'], random_state)

if algorithms['ACO']:
    aco_params = params.get('aco', {})
    aco = ACO(
        n_components=params['n_components'],
        n_ants=aco_params.get('n_ants', 30),
        n_iterations=aco_params.get('n_iterations', 100),
        evaporation_rate=aco_params.get('evaporation_rate', 0.1),
        random_state=random_state
    )
    X_aco = aco.fit_transform(X)
    
    reduced_data['ACO'] = {
        'embedding': X_aco,
        'labels': y,
        'target_names': target_names
    }
    
    metrics['ACO'] = calculate_metrics(X, X_aco, params['n_clusters'], random_state)

if algorithms['SOM']:
    som_params = params.get('som', {})
    som = SOMReducer(
        n_components=params['n_components'],
        sigma=som_params.get('sigma', 1.0),
        learning_rate=som_params.get('learning_rate', 0.5),
        n_iterations=som_params.get('n_iterations', 1000),
        map_size=(som_params.get('grid_size', 7), som_params.get('grid_size', 7)),
        random_state=random_state
    )
    X_som = som.fit_transform(X)
    
    reduced_data['SOM'] = {
        'embedding': X_som,
        'labels': y,
        'target_names': target_names
    }
    
    metrics['SOM'] = calculate_metrics(X, X_som, params['n_clusters'], random_state)

if algorithms['PSO']:
    pso_params = params.get('pso', {})
    pso = PSOReducer(
        n_components=params['n_components'],
        pop_size=pso_params.get('pop_size', 90),
        max_iter=pso_params.get('max_iter', 90),
        w=pso_params.get('w', 0.8),
        c1=pso_params.get('c1', 1.2),
        c2=pso_params.get('c2', 1.2),
        n_clusters=params['n_clusters'],
        random_state=random_state
    )
    X_pso = pso.fit_transform(X)
    
    reduced_data['PSO'] = {
        'embedding': X_pso,
        'labels': y,
        'target_names': target_names
    }
    
    metrics['PSO'] = calculate_metrics(X, X_pso, params['n_clusters'], random_state)

if algorithms['BAT']:
    bat_params = params.get('bat', {})
    bat = BATReducer(
        n_components=params['n_components'],
        n_bats=bat_params.get('n_bats', 40),
        max_iter=bat_params.get('max_iter', 100),
        f_min=bat_params.get('f_min', 0.0),
        f_max=bat_params.get('f_max', 2.0),
        alpha=bat_params.get('alpha', 0.9),
        gamma=bat_params.get('gamma', 0.9),
        random_state=random_state
    )
    X_bat = bat.fit_transform(X)
    
    reduced_data['BAT'] = {
        'embedding': X_bat,
        'labels': y,
        'target_names': target_names
    }
    
    metrics['BAT'] = calculate_metrics(X, X_bat, params['n_clusters'], random_state)

if algorithms['ABC']:
    abc_params = params.get('abc', {})
    abc = ABCReducer(
        n_components=params['n_components'],
        pop_size=abc_params.get('pop_size', 30),
        max_iter=abc_params.get('max_iter', 100),
        limit=abc_params.get('limit', 10),
        n_clusters=params['n_clusters'],
        random_state=random_state
    )
    X_abc = abc.fit_transform(X)
    
    reduced_data['ABC'] = {
        'embedding': X_abc,
        'labels': y,
        'target_names': target_names
    }
    
    metrics['ABC'] = calculate_metrics(X, X_abc, params['n_clusters'], random_state)

if algorithms['FA']:
    fa_params = params.get('fa', {})
    fa = FAReducer(
        n_components=params['n_components'],
        pop_size=fa_params.get('pop_size', 30),
        max_iter=fa_params.get('max_iter', 100),
        alpha=fa_params.get('alpha', 0.5),
        beta0=fa_params.get('beta0', 1.0),
        gamma=fa_params.get('gamma', 1.0),
        n_clusters=params['n_clusters'],
        random_state=random_state
    )
    X_fa = fa.fit_transform(X)
    
    reduced_data['FA'] = {
        'embedding': X_fa,
        'labels': y,
        'target_names': target_names
    }
    
    metrics['FA'] = calculate_metrics(X, X_fa, params['n_clusters'], random_state)

# Create tabs
tab1, tab2 = st.tabs(["📈 Visualizations", "📊 Comparison Metrics"])

# Fill tabs with content
with tab1:
    create_visualization_tab(algorithms, reduced_data, metrics)

with tab2:
    create_metrics_tab(metrics) 