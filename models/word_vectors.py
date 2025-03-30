import gensim.downloader as api
import streamlit as st

# Available models dictionary
MODEL_OPTIONS = {
    "GloVe (Wikipedia, 100d)": "glove-wiki-gigaword-100",
    "GloVe (Wikipedia, 300d)": "glove-wiki-gigaword-300",
    "Word2Vec (Google News, 300d)": "word2vec-google-news-300",
    "FastText (Wikipedia, 300d)": "fasttext-wiki-news-subwords-300"
}

@st.cache_resource
def load_model(model_name="glove-wiki-gigaword-100"):
    """
    Load a pre-trained word embedding model using Gensim's downloader API.
    The function is cached to prevent reloading the model on each rerun.
    
    Args:
        model_name (str): Name of the pre-trained model to load
        
    Returns:
        KeyedVectors: Word vectors from the loaded model
    """
    return api.load(model_name)

def get_word_vector(wv, word):
    """
    Get the vector representation of a word.
    
    Args:
        wv: Word vectors model
        word (str): The word to get the vector for
        
    Returns:
        numpy.ndarray: Vector representation of the word
    """
    if word in wv:
        return wv[word]
    return None

def find_similar_words(wv, word, topn=10):
    """
    Find words similar to the given word based on vector similarity.
    
    Args:
        wv: Word vectors model
        word (str): The word to find similar words for
        topn (int): Number of similar words to return
        
    Returns:
        list: List of (word, similarity) tuples
    """
    if word in wv:
        return wv.most_similar(word, topn=topn)
    return []

def get_word_similarity(wv, word1, word2):
    """
    Calculate similarity between two words.
    
    Args:
        wv: Word vectors model
        word1 (str): First word
        word2 (str): Second word
        
    Returns:
        float: Similarity score between the two words
    """
    if word1 in wv and word2 in wv:
        return wv.similarity(word1, word2)
    return None

def get_word_analogy(wv, word1, word2, word3, topn=5):
    """
    Solve word analogy: word1 is to ? as word2 is to word3.
    
    Args:
        wv: Word vectors model
        word1 (str): First word in the analogy
        word2 (str): Second word in the analogy
        word3 (str): Third word in the analogy
        topn (int): Number of results to return
        
    Returns:
        list: List of (word, similarity) tuples
    """
    if all(word in wv for word in [word1, word2, word3]):
        return wv.most_similar(positive=[word1, word3], negative=[word2], topn=topn)
    return []

def save_word_vectors(wv, output_file="word_vectors.kv"):
    """
    Save word vectors to a file for later use.
    
    Args:
        wv: Word vectors model
        output_file (str): Path to save the vectors
    """
    wv.save(output_file)
    return output_file