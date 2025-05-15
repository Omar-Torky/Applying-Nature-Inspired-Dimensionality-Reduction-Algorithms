import streamlit as st
from app.utils.visualization import create_scatter_plot

def create_visualization_tab(algorithms, data, metrics):
    """Create the visualization tab with plots for each algorithm."""
    st.markdown("<h2 class='sub-header'>2D Visualizations</h2>", unsafe_allow_html=True)
    st.write("Compare how different algorithms visualize the dataset in 2D space.")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_algorithms = sum(algorithms.values())
    current_progress = 0
    
    for algo_name, algo_data in data.items():
        if algorithms.get(algo_name, False):
            status_text.text(f"Showing {algo_name} visualization...")
            
            # Show plot
            st.plotly_chart(create_scatter_plot(
                algo_data['embedding'], 
                algo_data['labels'], 
                f"{algo_name} Dimensionality Reduction",
                algo_data['target_names']
            ))
            
            current_progress += 1
            progress_bar.progress(current_progress / total_algorithms)
    
    status_text.empty()

def create_metrics_tab(metrics):
    """Create the metrics comparison tab."""
    st.markdown("<h2 class='sub-header'>Algorithm Metrics Comparison</h2>", unsafe_allow_html=True)
    
    if metrics:
        for algo_name, metric_values in metrics.items():
            st.markdown(f"### {algo_name}")
            cols = st.columns(len(metric_values))
            
            for col, (metric_name, value) in zip(cols, metric_values.items()):
                with col:
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <h4>{metric_name.replace('_', ' ').title()}</h4>
                            <p>{value:.4f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    ) 