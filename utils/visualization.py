import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import altair as alt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_words(wv, words, method='pca'):
    """
    Visualize word vectors in 2D space using dimensionality reduction.
    
    Args:
        wv: Word vectors model
        words (list): List of words to visualize
        method (str): Dimensionality reduction method ('pca' or 'tsne')
        
    Returns:
        pandas.DataFrame: DataFrame with x, y coordinates and word labels
    """
    # Get word vectors
    word_vectors = [wv[word] for word in words if word in wv]
    if not word_vectors:
        st.error("None of the provided words are in the vocabulary!")
        return None
    
    # Get labels for the vectors
    labels = [word for word in words if word in wv]
    
    # Convert to array
    X = np.array(word_vectors)
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(len(words)-1, 30) if len(words) > 10 else 3)
    
    # Reduce dimensions
    X_reduced = reducer.fit_transform(X)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'word': labels
    })
    
    return df

def create_word_similarity_chart(results_df, orientation='h'):
    """
    Create a bar chart for word similarity results.
    
    Args:
        results_df (pandas.DataFrame): DataFrame with Word and Similarity columns
        orientation (str): Chart orientation ('h' for horizontal, 'v' for vertical)
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Reverse order for better visualization
    chart_df = results_df.iloc[::-1]
    
    fig = px.bar(
        chart_df, 
        x="Similarity" if orientation == 'h' else "Word", 
        y="Word" if orientation == 'h' else "Similarity", 
        orientation=orientation,
        title="Word Similarity Results",
        color="Similarity",
        color_continuous_scale="Blues"
    )
    
    return fig

def create_scatter_plot(plot_df, highlight_word=None, title="Word Vector Visualization"):
    """
    Create a scatter plot for word vector visualization.
    
    Args:
        plot_df (pandas.DataFrame): DataFrame with x, y coordinates and word labels
        highlight_word (str): Word to highlight in the plot
        title (str): Plot title
        
    Returns:
        altair.Chart: Altair chart object
    """
    if highlight_word:
        plot_df['is_highlighted'] = plot_df['word'] == highlight_word
        
        chart = alt.Chart(plot_df).mark_circle(size=100).encode(
            x='x',
            y='y',
            color=alt.condition(
                alt.datum.is_highlighted,
                alt.value('red'),
                alt.value('blue')
            ),
            tooltip=['word']
        ).properties(
            width=600,
            height=400,
            title=title
        )
    else:
        chart = alt.Chart(plot_df).mark_circle(size=100).encode(
            x='x',
            y='y',
            tooltip=['word']
        ).properties(
            width=600,
            height=400,
            title=title
        )
    
    # Add text labels
    text = alt.Chart(plot_df).mark_text(dx=15, dy=3).encode(
        x='x',
        y='y',
        text='word'
    )
    
    return chart + text

def create_analogy_visualization(plot_df, word1, word2, word3, result_word):
    """
    Create a visualization for word analogies.
    
    Args:
        plot_df (pandas.DataFrame): DataFrame with x, y coordinates and word labels
        word1, word2, word3 (str): Words in the analogy
        result_word (str): Result of the analogy
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = px.scatter(
        plot_df, 
        x="x", 
        y="y", 
        text="word",
        title="2D Projection of Analogy Vectors"
    )
    
    # Add lines to connect the analogy pairs
    fig.add_shape(type="line", x0=plot_df[plot_df['word']==word2]['x'].values[0], 
                 y0=plot_df[plot_df['word']==word2]['y'].values[0],
                 x1=plot_df[plot_df['word']==word3]['x'].values[0],
                 y1=plot_df[plot_df['word']==word3]['y'].values[0],
                 line=dict(color="Blue", width=2, dash="dash"))
    
    fig.add_shape(type="line", x0=plot_df[plot_df['word']==word1]['x'].values[0], 
                 y0=plot_df[plot_df['word']==word1]['y'].values[0],
                 x1=plot_df[plot_df['word']==result_word]['x'].values[0],
                 y1=plot_df[plot_df['word']==result_word]['y'].values[0],
                 line=dict(color="Red", width=2, dash="dash"))
    
    fig.update_traces(textposition='top center')
    fig.update_layout(showlegend=False)
    
    return fig