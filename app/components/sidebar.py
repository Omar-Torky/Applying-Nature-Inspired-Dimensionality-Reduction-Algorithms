import streamlit as st

def create_sidebar():
    """Create the sidebar with algorithm selection and parameters."""
    st.sidebar.title("Settings")
    
    # Visualization settings
    st.sidebar.markdown("### Visualization Settings")
    viz_params = {
        'show_3d': st.sidebar.checkbox("Show 3D Visualization", value=False),
        'show_umatrix': st.sidebar.checkbox("Show U-Matrix (for SOM)", value=True)
    }
    
    # Algorithm selection
    st.sidebar.markdown("### Choose Algorithms to Compare")
    algorithms = {
        'PCA': st.sidebar.checkbox("PCA", value=True),
        't-SNE': st.sidebar.checkbox("t-SNE", value=True),
        'ACO': st.sidebar.checkbox("Ant Colony Optimization (ACO)", value=False),
        'SOM': st.sidebar.checkbox("Self-Organizing Maps (SOM)", value=False),
        'PSO': st.sidebar.checkbox("Particle Swarm Optimization (PSO)", value=False),
        'ABC': st.sidebar.checkbox("Artificial Bee Colony (ABC)", value=False),
        'BAT': st.sidebar.checkbox("Bat Algorithm", value=False),
        'FA': st.sidebar.checkbox("Firefly Algorithm", value=False)
    }
    
    # General parameters
    st.sidebar.markdown("### General Parameters")
    params = {
        'random_state': st.sidebar.slider("Random State", 0, 100, 42),
        'n_clusters': st.sidebar.slider("Number of Clusters for Metrics", 2, 10, 3),
        'n_components': 3 if viz_params['show_3d'] else 2
    }
    
    # Algorithm-specific parameters
    if algorithms['ACO']:
        st.sidebar.markdown("### ACO Parameters")
        params['aco'] = {
            'n_ants': st.sidebar.slider("Number of Ants", 10, 100, 30),
            'n_iterations': st.sidebar.slider("Number of Iterations", 50, 500, 100),
            'evaporation_rate': st.sidebar.slider("Evaporation Rate", 0.0, 1.0, 0.1)
        }
    
    if algorithms['SOM']:
        st.sidebar.markdown("### SOM Parameters")
        params['som'] = {
            'sigma': st.sidebar.slider("Sigma", 0.1, 5.0, 1.0),
            'learning_rate': st.sidebar.slider("Learning Rate", 0.1, 1.0, 0.5),
            'n_iterations': st.sidebar.slider("Number of Iterations", 100, 2000, 1000),
            'grid_size': st.sidebar.slider("Grid Size", 5, 15, 7)
        }
    
    if algorithms['PSO']:
        st.sidebar.markdown("### PSO Parameters")
        params['pso'] = {
            'pop_size': st.sidebar.slider("Population Size (PSO)", 30, 150, 90),
            'max_iter': st.sidebar.slider("Max Iterations (PSO)", 50, 200, 90),
            'w': st.sidebar.slider("Inertia Weight", 0.1, 1.0, 0.8),
            'c1': st.sidebar.slider("Cognitive Parameter", 0.1, 2.0, 1.2),
            'c2': st.sidebar.slider("Social Parameter", 0.1, 2.0, 1.2)
        }
    
    if algorithms['BAT']:
        st.sidebar.markdown("### BAT Parameters")
        params['bat'] = {
            'n_bats': st.sidebar.slider("Number of Bats", 20, 100, 40),
            'max_iter': st.sidebar.slider("Max Iterations (BAT)", 50, 200, 100),
            'f_min': st.sidebar.slider("Minimum Frequency", 0.0, 1.0, 0.0),
            'f_max': st.sidebar.slider("Maximum Frequency", 1.0, 5.0, 2.0),
            'alpha': st.sidebar.slider("Alpha (BAT)", 0.1, 1.0, 0.9),
            'gamma': st.sidebar.slider("Gamma (BAT)", 0.1, 1.0, 0.9)
        }
    
    if algorithms['ABC']:
        st.sidebar.markdown("### ABC Parameters")
        params['abc'] = {
            'pop_size': st.sidebar.slider("Colony Size", 20, 100, 30),
            'max_iter': st.sidebar.slider("Max Iterations (ABC)", 50, 200, 100),
            'limit': st.sidebar.slider("Limit", 5, 20, 10)
        }
        
    if algorithms['FA']:
        st.sidebar.markdown("### Firefly Parameters")
        params['fa'] = {
            'pop_size': st.sidebar.slider("Number of Fireflies", 20, 100, 30),
            'max_iter': st.sidebar.slider("Max Iterations (FA)", 50, 200, 100),
            'alpha': st.sidebar.slider("Alpha (FA)", 0.1, 1.0, 0.5),
            'beta0': st.sidebar.slider("Beta0", 0.1, 2.0, 1.0),
            'gamma': st.sidebar.slider("Gamma (FA)", 0.1, 2.0, 1.0)
        }
    
    return algorithms, params, viz_params 