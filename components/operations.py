import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt

from utils.visualization import (
    visualize_words, 
    create_word_similarity_chart, 
    create_scatter_plot,
    create_analogy_visualization
)

def render_similar_words(wv):
    """Render the UI for finding similar words"""
    st.markdown("<h2 class='subheader'>Find Similar Words</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        word = st.text_input("Enter a word:", "king")
        num_results = st.slider("Number of similar words:", 1, 50, 10)
    
    with col2:
        st.markdown("<div class='info-box'>This tool finds words with similar meanings based on their vector representations.</div>", unsafe_allow_html=True)
        chart_type = st.selectbox("Display Results As:", ["Table", "Bar Chart", "Scatter Plot"])
    
    if st.button("Find Similar Words", key="find_similar"):
        try:
            similar_words = wv.most_similar(word, topn=num_results)
            
            # Create DataFrame for results
            results_df = pd.DataFrame(similar_words, columns=["Word", "Similarity"])
            
            if chart_type == "Table":
                st.dataframe(results_df.style.highlight_max(subset=["Similarity"]))
            
            elif chart_type == "Bar Chart":
                # Reverse order for better visualization
                chart_df = results_df.iloc[::-1]
                fig = px.bar(
                    chart_df, 
                    x="Similarity", 
                    y="Word", 
                    orientation='h',
                    title=f"Words Similar to '{word}'",
                    color="Similarity",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:  # Scatter Plot
                # Add the original word to visualization
                words_to_plot = [word] + [w for w, _ in similar_words]
                plot_df = visualize_words(wv, words_to_plot)
                
                if plot_df is not None:
                    # Highlight the original word
                    plot_df['is_original'] = plot_df['word'] == word
                    
                    chart = alt.Chart(plot_df).mark_circle(size=100).encode(
                        x='x',
                        y='y',
                        color=alt.condition(
                            alt.datum.is_original,
                            alt.value('red'),
                            alt.value('blue')
                        ),
                        tooltip=['word']
                    ).properties(
                        width=600,
                        height=400,
                        title=f"2D Projection of '{word}' and Similar Words"
                    )
                    
                    # Add text labels
                    text = alt.Chart(plot_df).mark_text(dx=15, dy=3).encode(
                        x='x',
                        y='y',
                        text='word'
                    )
                    
                    st.altair_chart(chart + text, use_container_width=True)
            
        except KeyError:
            st.error(f"Word '{word}' not found in vocabulary!")

def render_word_similarity(wv):
    """Render the UI for comparing word similarity"""
    st.markdown("<h2 class='subheader'>Compare Word Similarity</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        word1 = st.text_input("Enter first word:", "apple")
    with col2:
        word2 = st.text_input("Enter second word:", "orange")
    
    # Add option for multiple word pairs
    st.markdown("### Compare Multiple Word Pairs")
    additional_pairs = st.checkbox("Add more word pairs for comparison")
    
    word_pairs = [(word1, word2)]
    
    if additional_pairs:
        num_pairs = st.number_input("Number of additional pairs:", 1, 5, 2)
        
        for i in range(num_pairs):
            col1, col2 = st.columns(2)
            with col1:
                w1 = st.text_input(f"Word {i+1}A:", f"example{i+1}a")
            with col2:
                w2 = st.text_input(f"Word {i+1}B:", f"example{i+1}b")
            word_pairs.append((w1, w2))
    
    if st.button("Check Similarity", key="check_sim"):
        results = []
        
        for w1, w2 in word_pairs:
            try:
                similarity = wv.similarity(w1, w2)
                results.append({
                    "Word 1": w1,
                    "Word 2": w2,
                    "Similarity": similarity
                })
                
                # Display individual result for the main pair
                if w1 == word1 and w2 == word2:
                    st.metric(label=f"Similarity between '{w1}' and '{w2}'", value=f"{similarity:.4f}")
                    
                    # Visualization of similarity
                    if similarity >= 0.7:
                        st.success("These words are very similar!")
                    elif similarity >= 0.4:
                        st.info("These words have moderate similarity.")
                    else:
                        st.warning("These words are not very similar.")
                    
                    # Show common similar words
                    st.markdown("### Common Similar Words")
                    try:
                        similar_to_w1 = set([word for word, _ in wv.most_similar(w1, topn=20)])
                        similar_to_w2 = set([word for word, _ in wv.most_similar(w2, topn=20)])
                        common_words = similar_to_w1.intersection(similar_to_w2)
                        
                        if common_words:
                            st.write(f"Words similar to both '{w1}' and '{w2}':")
                            st.write(", ".join(common_words))
                        else:
                            st.info("No common similar words found in the top 20.")
                    except KeyError:
                        pass
                    
            except KeyError:
                st.error(f"One or both words ('{w1}', '{w2}') not found in vocabulary!")
        
        # Display comparison table if multiple pairs
        if len(results) > 1:
            st.markdown("### Comparison of All Word Pairs")
            results_df = pd.DataFrame(results)
            
            # Create bar chart
            fig = px.bar(
                results_df, 
                x="Similarity", 
                y=[f"{r['Word 1']} - {r['Word 2']}" for r in results],
                orientation='h',
                title="Similarity Comparison",
                color="Similarity",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig, use_container_width=True)

def render_word_analogy(wv):
    """Render the UI for word analogies"""
    st.markdown("<h2 class='subheader'>Word Analogy Solver</h2>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>This tool solves analogies like: king is to queen as man is to woman</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        word1 = st.text_input("Word 1:", "king")
    with col2:
        word2 = st.text_input("Word 2:", "man")
    with col3:
        word3 = st.text_input("Word 3:", "woman")
    
    num_results = st.slider("Number of results:", 1, 10, 3)
    
    st.markdown(f"### Formula: {word1} is to ? as {word2} is to {word3}")
    
    if st.button("Find Analogous Word", key="analogy"):
        try:
            results = wv.most_similar(positive=[word1, word3], negative=[word2], topn=num_results)
            
            # Create DataFrame for results
            results_df = pd.DataFrame(results, columns=["Word", "Score"])
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Results:")
                st.dataframe(results_df)
                
                # Create visualization
                fig = px.bar(
                    results_df,
                    x="Score",
                    y="Word",
                    orientation='h',
                    title=f"Top {num_results} Analogous Words",
                    color="Score",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig)
            
            with col2:
                st.subheader("Explanation:")
                st.markdown(f"""
                The analogy "{word1} is to ? as {word2} is to {word3}" is solved by:
                
                1. Finding the vector difference: {word3} - {word2}
                2. Adding this difference to {word1}
                3. Finding words closest to this new vector
                
                The top result suggests that **{results[0][0]}** is to **{word1}** as **{word3}** is to **{word2}**.
                """)
                
                # Visualize the analogy
                words_to_plot = [word1, word2, word3, results[0][0]]
                plot_df = visualize_words(wv, words_to_plot)
                
                if plot_df is not None:
                    st.subheader("Vector Visualization:")
                    
                    # Use the utility function for analogy visualization
                    fig = create_analogy_visualization(plot_df, word1, word2, word3, results[0][0])
                    st.plotly_chart(fig, use_container_width=True)
                
        except KeyError:
            st.error("One or more words not found in vocabulary!")

def render_word_clustering(wv):
    """Render the UI for word clustering"""
    st.markdown("<h2 class='subheader'>Word Clustering</h2>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Explore how words cluster together in the embedding space</div>", unsafe_allow_html=True)
    
    # Input for words
    input_method = st.radio("Input Method:", ["Enter Words", "Choose Category"])
    
    words_to_cluster = []
    
    if input_method == "Enter Words":
        word_input = st.text_area("Enter words (one per line or comma-separated):", "king\nqueen\nman\nwoman\nprince\nprincess\nboy\ngirl")
        
        if word_input:
            # Parse input (handle both comma-separated and line-separated)
            if ',' in word_input:
                words_to_cluster = [w.strip() for w in word_input.split(',')]
            else:
                words_to_cluster = [w.strip() for w in word_input.split('\n')]
            
            # Remove empty strings
            words_to_cluster = [w for w in words_to_cluster if w]
    
    else:  # Choose Category
        categories = {
            "Countries": "usa,canada,france,germany,italy,spain,china,japan,india,brazil,russia,australia",
            "Animals": "dog,cat,horse,cow,lion,tiger,elephant,giraffe,monkey,bear,wolf,fox",
            "Colors": "red,blue,green,yellow,orange,purple,black,white,brown,pink,gray,violet",
            "Food": "apple,banana,orange,pizza,pasta,bread,rice,cheese,meat,fish,vegetable,fruit",
            "Technology": "computer,phone,internet,software,hardware,data,algorithm,program,code,app,website,digital"
        }
        
        selected_category = st.selectbox("Select a category:", list(categories.keys()))
        words_to_cluster = [w.strip() for w in categories[selected_category].split(',')]
        
        st.write(f"Words in category: {', '.join(words_to_cluster)}")
    
    # Visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        viz_method = st.selectbox("Visualization Method:", ["PCA", "t-SNE"])
    
    with col2:
        label_size = st.slider("Label Size:", 8, 20, 12)
    
    if st.button("Generate Cluster Visualization", key="cluster"):
        if len(words_to_cluster) < 3:
            st.error("Please enter at least 3 words for meaningful clustering.")
        else:
            # Filter words that are in vocabulary
            valid_words = [w for w in words_to_cluster if w in wv]
            invalid_words = set(words_to_cluster) - set(valid_words)
            
            if invalid_words:
                st.warning(f"The following words were not found in the vocabulary: {', '.join(invalid_words)}")
            
            if valid_words:
                # Get 2D representation
                plot_df = visualize_words(wv, valid_words, method='pca' if viz_method == 'PCA' else 'tsne')
                
                if plot_df is not None:
                    st.subheader(f"{viz_method} Visualization of Word Clusters")
                    
                    # Create interactive scatter plot
                    fig = px.scatter(
                        plot_df, 
                        x="x", 
                        y="y", 
                        text="word",
                        title=f"{viz_method} Projection of Word Embeddings"
                    )
                    
                    fig.update_traces(textposition='top center', textfont=dict(size=label_size))
                    fig.update_layout(showlegend=False)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add option to download the coordinates
                    csv = plot_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Visualization Data",
                        csv,
                        "word_clusters.csv",
                        "text/csv",
                        key='download-clusters'
                    )

def render_embedding_visualization(wv):
    """Render the UI for embedding visualization"""
    st.markdown("<h2 class='subheader'>Embedding Visualization</h2>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Explore the relationships between words in the embedding space</div>", unsafe_allow_html=True)
    
    # Word vector operations
    st.subheader("Word Vector Operations")
    
    operation = st.selectbox(
        "Select Operation:",
        ["Addition (A + B)", "Subtraction (A - B)", "Custom (αA + βB - γC)"]
    )
    
    if operation == "Addition (A + B)":
        col1, col2 = st.columns(2)
        with col1:
            word_a = st.text_input("Word A:", "king")
        with col2:
            word_b = st.text_input("Word B:", "woman")
            
        if st.button("Compute A + B", key="add_op"):
            try:
                # Compute the result vector
                result_vector = wv[word_a] + wv[word_b]
                
                # Find closest words to the result
                result_words = wv.similar_by_vector(result_vector, topn=5)
                
                st.success(f"Result of {word_a} + {word_b}:")
                
                # Display results
                for word, score in result_words:
                    st.write(f"**{word}**: {score:.4f}")
                
                # Visualize
                words_to_plot = [word_a, word_b] + [w for w, _ in result_words]
                plot_df = visualize_words(wv, words_to_plot)
                
                if plot_df is not None:
                    # Mark the input words differently
                    plot_df['type'] = 'result'
                    plot_df.loc[plot_df['word'].isin([word_a, word_b]), 'type'] = 'input'
                    
                    fig = px.scatter(
                        plot_df, 
                        x="x", 
                        y="y", 
                        color="type",
                        text="word",
                        title=f"Visualization of {word_a} + {word_b}",
                        color_discrete_map={'input': 'red', 'result': 'blue'}
                    )
                    
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)
                
            except KeyError as e:
                st.error(f"Word not found in vocabulary: {str(e)}")
    
    elif operation == "Subtraction (A - B)":
        col1, col2 = st.columns(2)
        with col1:
            word_a = st.text_input("Word A:", "king")
        with col2:
            word_b = st.text_input("Word B:", "man")
            
        if st.button("Compute A - B", key="sub_op"):
            try:
                # Compute the result vector
                result_vector = wv[word_a] - wv[word_b]
                
                # Find closest words to the result
                result_words = wv.similar_by_vector(result_vector, topn=5)
                
                st.success(f"Result of {word_a} - {word_b}:")
                
                # Display results
                for word, score in result_words:
                    st.write(f"**{word}**: {score:.4f}")
                
                # Visualize
                words_to_plot = [word_a, word_b] + [w for w, _ in result_words]
                plot_df = visualize_words(wv, words_to_plot)
                
                if plot_df is not None:
                    # Mark the input words differently
                    plot_df['type'] = 'result'
                    plot_df.loc[plot_df['word'].isin([word_a, word_b]), 'type'] = 'input'
                    
                    fig = px.scatter(
                        plot_df, 
                        x="x", 
                        y="y", 
                        color="type",
                        text="word",
                        title=f"Visualization of {word_a} - {word_b}",
                        color_discrete_map={'input': 'red', 'result': 'blue'}
                    )
                    
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)
                
            except KeyError as e:
                st.error(f"Word not found in vocabulary: {str(e)}")
    
    else:  # Custom operation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            word_a = st.text_input("Word A:", "king")
            alpha = st.slider("α (Weight for A):", 0.0, 2.0, 1.0, 0.1)
        
        with col2:
            word_b = st.text_input("Word B:", "woman")
            beta = st.slider("β (Weight for B):", 0.0, 2.0, 1.0, 0.1)
        
        with col3:
            word_c = st.text_input("Word C:", "man")
            gamma = st.slider("γ (Weight for C):", 0.0, 2.0, 1.0, 0.1)
        
        st.markdown(f"### Formula: {alpha}×{word_a} + {beta}×{word_b} - {gamma}×{word_c}")
        
        if st.button("Compute Custom Operation", key="custom_op"):
            try:
                # Compute the result vector
                result_vector = alpha * wv[word_a] + beta * wv[word_b] - gamma * wv[word_c]
                
                # Find closest words to the result
                result_words = wv.similar_by_vector(result_vector, topn=5)
                
                st.success(f"Result of {alpha}×{word_a} + {beta}×{word_b} - {gamma}×{word_c}:")
                
                # Display results
                for word, score in result_words:
                    st.write(f"**{word}**: {score:.4f}")
                
                # Visualize
                words_to_plot = [word_a, word_b, word_c] + [w for w, _ in result_words]
                plot_df = visualize_words(wv, words_to_plot)
                
                if plot_df is not None:
                    # Mark the input words differently
                    plot_df['type'] = 'result'
                    plot_df.loc[plot_df['word'].isin([word_a, word_b, word_c]), 'type'] = 'input'
                    
                    fig = px.scatter(
                        plot_df, 
                        x="x", 
                        y="y", 
                        color="type",
                        text="word",
                        title=f"Visualization of Custom Operation",
                        color_discrete_map={'input': 'red', 'result': 'blue'}
                    )
                    
                    fig.update_traces(textposition='top center')
                    st.plotly_chart(fig, use_container_width=True)
                
            except KeyError as e:
                st.error(f"Word not found in vocabulary: {str(e)}")