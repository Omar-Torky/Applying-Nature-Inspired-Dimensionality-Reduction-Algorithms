import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def create_scatter_plot(X_reduced, y, title, target_names, is_3d=False):
    """Create a Plotly scatter plot for 2D or 3D visualization."""
    if is_3d and X_reduced.shape[1] >= 3:
        df = pd.DataFrame({
            'x': X_reduced[:, 0],
            'y': X_reduced[:, 1],
            'z': X_reduced[:, 2],
            'class': [target_names[label] for label in y]
        })
        
        fig = px.scatter_3d(
            df, x='x', y='y', z='z', color='class',
            color_discrete_sequence=px.colors.qualitative.Set1,
            title=title,
            labels={
                'x': 'Component 1',
                'y': 'Component 2',
                'z': 'Component 3',
                'class': 'Wine Variety'
            }
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            height=600,
            legend_title_text='Wine Variety',
            font=dict(family="Arial", size=12),
            title=dict(text=title, x=0.5)
        )
    else:
        df = pd.DataFrame({
            'x': X_reduced[:, 0],
            'y': X_reduced[:, 1],
            'class': [target_names[label] for label in y]
        })
        
        fig = px.scatter(
            df, x='x', y='y', color='class',
            color_discrete_sequence=px.colors.qualitative.Set1,
            title=title,
            labels={'x': 'Component 1', 'y': 'Component 2', 'class': 'Wine Variety'}
        )
        
        fig.update_layout(
            height=400,
            legend_title_text='Wine Variety',
            font=dict(family="Arial", size=12),
            title=dict(text=title, x=0.5)
        )
    
    return fig

def create_u_matrix_plot(u_matrix, title="U-Matrix"):
    """Create a heatmap plot for U-Matrix visualization."""
    fig = go.Figure(data=go.Heatmap(
        z=u_matrix,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        height=400,
        xaxis_title="SOM Grid X",
        yaxis_title="SOM Grid Y"
    )
    
    return fig 