import streamlit as st

# Import from our modules
from models.word_vectors import load_model, MODEL_OPTIONS
from components.sidebar import render_sidebar
from components.operations import (
    render_similar_words,
    render_word_similarity,
    render_word_analogy,
    render_word_clustering,
    render_embedding_visualization
)

# Set page configuration
st.set_page_config(
    page_title="Word Embedding Explorer",
    page_icon="🔤",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main Header - Bold & Eye-catching */
    .main-header {
        font-size: 2.5rem;
        color: #1565C0; /* Deep Blue */
        text-align: center;
        font-weight: bold;
    }

    /* Subheader - Slightly Softer but Strong */
    .subheader {
        font-size: 1.5rem;
        color: #1E88E5; /* Medium Blue */
        font-weight: 600;
    }

    /* Info Box - Light and Easy on the Eyes */
    .info-box {
        background-color: #BBDEFB; /* Soft Sky Blue */
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #1E88E5; /* Adds a subtle accent */
    }

    /* Metric Box - Soft & Elegant */
    .stMetric {
        background-color: #C5CAE9; /* Light Indigo */
        padding: 10px;
        border-radius: 5px;
        color: #283593; /* Darker contrast text */
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main content area
st.markdown(f"<h1 class='main-header'>Advanced Word Embedding Explorer</h1>", unsafe_allow_html=True)

# Load model and render sidebar
model_name, wv, option = render_sidebar()

# Different operations
if option == "Find Similar Words":
    render_similar_words(wv)
elif option == "Word Similarity":
    render_word_similarity(wv)
elif option == "Word Analogy":
    render_word_analogy(wv)
elif option == "Word Clustering":
    render_word_clustering(wv)
elif option == "Embedding Visualization":
    render_embedding_visualization(wv)
