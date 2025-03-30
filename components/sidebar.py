import streamlit as st
import pandas as pd
from models.word_vectors import MODEL_OPTIONS, load_model

def render_sidebar(wv=None):
    """
    Render the sidebar UI components.
    
    Args:
        wv: Word vectors model (optional, will be loaded if not provided)
        
    Returns:
        tuple: (selected_model_name, word_vectors_model, selected_operation)
    """
    st.sidebar.title("Word Embedding Explorer")
    st.sidebar.markdown("---")
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "Select Word Embedding Model:",
        list(MODEL_OPTIONS.keys())
    )
    
    # Load the selected model
    model_name = MODEL_OPTIONS[selected_model]
    if wv is None:
        with st.spinner(f"Loading {selected_model}..."):
            wv = load_model(model_name)
            st.sidebar.success("Model loaded successfully!")
    
    st.sidebar.markdown("---")
    
    # Operation selection
    operation = st.sidebar.radio(
        "Choose an operation:", 
        ["Find Similar Words", "Word Similarity", "Word Analogy", "Word Clustering", "Embedding Visualization"]
    )
    
    # Display model info
    st.sidebar.markdown("### Model Information")
    st.sidebar.info(f"Vocabulary Size: {len(wv.key_to_index):,} words\nVector Dimensions: {wv.vector_size}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Created by Naman 🚀")
    st.sidebar.info(f"Using {selected_model} embeddings")
    
    # Add download option for vocabulary
    if st.sidebar.button("Download Model Vocabulary"):
        vocab_df = pd.DataFrame(list(wv.key_to_index.keys()), columns=["Word"])
        csv = vocab_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            "Click to Download",
            csv,
            "vocabulary.csv",
            "text/csv",
            key='download-vocab'
        )
    
    return model_name, wv, operation