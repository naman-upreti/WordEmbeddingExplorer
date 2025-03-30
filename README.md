# 🚀 **Word Embedding Explorer**  

**An interactive platform for exploring pre-trained word embeddings using Gensim, powered by a Streamlit web application.**  

---

## 📌 **Project Overview**  

🔹 **Word embeddings** are numerical vector representations of words that capture their semantic meanings. This project demonstrates how they can be used for:  
✅ **Finding similar words**  
✅ **Measuring word similarity**  
✅ **Solving word analogies**  
✅ **Clustering words based on meanings**  
✅ **Visualizing word embeddings in 2D space**  

---

## 🏗 **Project Structure**  

```
📂 Word Embedding Explorer
├── 📜 README.md               # Project documentation
├── 🖥 app.py                   # Streamlit web application (Main UI)
├── 📂 models                   # Model loading & management
│   ├── __init__.py           
│   └── word_vectors.py        # Pre-trained word vector model loader
├── 📂 utils                    # Utility functions
│   ├── __init__.py           
│   └── visualization.py       # Word embedding visualizations (PCA, t-SNE)
├── 📂 components               # UI components (Sidebar & Operations)
│   ├── __init__.py           
│   ├── sidebar.py             # Sidebar UI & model selection
│   └── operations.py          # Word similarity, analogy, clustering
├── 📓 Simple_Word2Vec_Project.ipynb  # Jupyter Notebook (Demo & Explanation)
└── 📜 requirements.txt        # Dependencies list
```

---

## ⚙ **Implementation Details**  

### 🔹 **1. Model Management (`models/word_vectors.py`)**  
✔ Loads **pre-trained word embedding models** (GloVe, Word2Vec, FastText).  
✔ Caches models to **speed up loading**.  
✔ Provides a unified API for accessing different embeddings.  

### 🔹 **2. Visualization Utilities (`utils/visualization.py`)**  
✔ **PCA & t-SNE** for reducing high-dimensional embeddings to 2D.  
✔ Interactive **scatter plots** to visualize word relationships.  

### 🔹 **3. UI Components (`components/sidebar.py & operations.py`)**  
✔ **Sidebar for model selection & operations.**  
✔ Implements **word similarity, analogy solving, clustering**, and other interactive tools.  

---

## 🎯 **Key Features**  

### 📍 **1. Find Similar Words**  
🔹 Input a word → Get **top similar words** based on vector similarity.  
🔹 Uses **cosine similarity** to find closest words.  

### 📍 **2. Word Similarity Check**  
🔹 Compare two words → See their **similarity score (-1 to 1)**.  
🔹 Uses `similarity()` from Gensim.  

### 📍 **3. Word Analogy Solver**  
🔹 Example: **"King - Man + Woman = ?"** → **"Queen"**  
🔹 Uses vector arithmetic to solve analogies.  

### 📍 **4. Word Clustering**  
🔹 Groups words with similar meanings **using K-Means clustering**.  
🔹 Helps visualize word relationships.  

### 📍 **5. Embedding Visualization**  
🔹 Reduces 300D vectors into 2D using **PCA / t-SNE**.  
🔹 Creates an **interactive word map**.  

---

## ⚡ **Tech Stack**  

| 🛠 Technology       | 🔹 Description |
|--------------------|---------------|
| ![Streamlit](https://img.shields.io/badge/🔴-Streamlit-FF4B4B?style=for-the-badge) | Web app framework for UI |
| ![Gensim](https://img.shields.io/badge/🟢-Gensim-1A9E6D?style=for-the-badge) | NLP library for embeddings |
| ![Pandas](https://img.shields.io/badge/🐼-Pandas-150458?style=for-the-badge) | Data handling & manipulation |
| ![NumPy](https://img.shields.io/badge/🔢-NumPy-013243?style=for-the-badge) | Matrix & vector operations |
| ![Matplotlib](https://img.shields.io/badge/📊-Matplotlib-11557C?style=for-the-badge) | Static data visualizations |
| ![Plotly](https://img.shields.io/badge/📈-Plotly-3F4F75?style=for-the-badge) | Interactive graphs & charts |
| ![Scikit-Learn](https://img.shields.io/badge/🤖-Scikit--Learn-F7931E?style=for-the-badge) | ML tools for PCA & t-SNE |
| ![Altair](https://img.shields.io/badge/📉-Altair-Red?style=for-the-badge) | Statistical visualizations |

---

## 📥 **Installation**  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/word-embedding-explorer.git
cd word-embedding-explorer
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit Web App  
```bash
streamlit run app.py
```

### 4️⃣ Open Jupyter Notebook (For Demo)  
```bash
jupyter notebook Simple_Word2Vec_Project.ipynb
```

---

## 📚 **Supported Pre-trained Models**  

| Model | Dataset | Dimensions |
|--------|---------|------------|
| GloVe | Wikipedia | 100d / 300d |
| Word2Vec | Google News | 300d |
| FastText | Wikipedia | 300d |

---

## 🔍 **Key Findings & Insights**  

✅ **Word embeddings capture deep semantic relationships.**  
✅ **Analogies can be solved through vector operations.**  
✅ **Different models have different strengths:**  
- **GloVe** excels at capturing global statistics.  
- **Word2Vec** focuses on local word relationships.  
- **FastText** handles rare words better with subword embeddings.  
✅ **Dimensionality reduction (PCA, t-SNE) reveals hidden clusters in language.**  

---

## 🚀 **Future Enhancements**  

🔹 **Integrate Transformer-based models (BERT, GPT).**  
🔹 **Support multilingual embeddings.**  
🔹 **Allow users to upload custom word vectors.**  
🔹 **Improve visualization with 3D embeddings.**  

---

## 📜 **License**  
This project is open-source under the **MIT License**.  

## 👤 **Author**  
📌 **Created by [Naman](https://github.com/yourusername) 🚀**  

---
