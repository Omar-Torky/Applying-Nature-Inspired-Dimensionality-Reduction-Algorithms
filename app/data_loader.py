import streamlit as st
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_data():
    """Load and preprocess the wine dataset."""
    data = load_wine()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    return X, y, feature_names, target_names

def get_dataset_info(X, y, target_names):
    """Return formatted dataset information."""
    info = {
        "Dataset": "Wine Dataset",
        "Samples": X.shape[0],
        "Features": X.shape[1],
        "Classes": len(np.unique(y)),
        "Class names": ", ".join(target_names)
    }
    return info 