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
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .subheader {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stMetric {
        background-color: #E8EAF6;
        padding: 10px;
        border-radius: 5px;
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
