import streamlit as st

def setup_page_config():
    st.set_page_config(
        page_title="Nature-Inspired Dimensionality Reduction",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #4CAF50;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #2196F3;
            margin-top: 2rem;
        }
        .algorithm-section {
            background-color: #f9f9f9;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
            margin: 0.5rem 0;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #757575;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True) 