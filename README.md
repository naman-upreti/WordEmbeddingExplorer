
# **GoogleNews Word2Vec - Pretrained Word Embeddings**

## 📌 **Introduction**
Word embeddings are numerical representations of words in a high-dimensional space. These embeddings capture semantic relationships between words and are widely used in **Natural Language Processing (NLP)** tasks.

The **GoogleNews-vectors-negative300.bin** file is a **pretrained Word2Vec model** trained on **Google News corpus (100 billion words)** and represents words as **300-dimensional vectors**. This model can be used for:
- **Finding similar words**
- **Word analogy tasks** (e.g., "king" → "man" as "queen" → "woman")
- **NLP applications** like sentiment analysis, text classification, and search engines.

---

## 🧠 **What is Word2Vec?**
Word2Vec is a deep learning-based technique that transforms words into vector representations. It uses two architectures:
1. **CBOW (Continuous Bag of Words)** – Predicts a word from surrounding words.
2. **Skip-gram** – Predicts surrounding words from a given word.

### 🎯 **Why Use Word2Vec?**
- **Captures word meanings** based on context.
- **Mathematical operations** on words (e.g., "Paris" - "France" + "Italy" = "Rome").
- **Improves NLP model performance** by understanding semantic similarity.

---

## ⚙ **How to Set Up and Use GoogleNews Word2Vec Model**
### 1️⃣ **Installation**
Ensure Python and required libraries are installed.

```bash
pip install gensim numpy
```

### 2️⃣ **Download the Pretrained Model**
The model file (~1.5GB) can be downloaded from:
- **Official Google Archive**: [GoogleNews Word2Vec](https://code.google.com/archive/p/word2vec/)

Move `GoogleNews-vectors-negative300.bin` to your working directory.

### 3️⃣ **Loading the Model in Python**
```python
import gensim

# Load the pre-trained Word2Vec model
model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# Check if the model is loaded
print("Model Loaded Successfully!")
```

---

## 🔥 **Hands-on Learning with Word2Vec**
### 📖 **1. Find Similar Words**
The model can find words similar to a given word based on cosine similarity.

```python
similar_words = model.most_similar("apple")
print(similar_words)
```

**Example Output:**
```
[('apples', 0.8501), ('pear', 0.8213), ('fruit', 0.8105), ('banana', 0.7986)]
```
**Explanation:**  
"Apple" is most similar to "apples," "pear," and other fruits.

---

### 🔄 **2. Word Analogies (Semantic Relationships)**
Word2Vec allows analogy-based word retrieval.

```python
result = model.most_similar(positive=['king', 'woman'], negative=['man'])
print(result)
```

**Example Output:**
```
[('queen', 0.8792)]
```
**Explanation:**  
_"King" is to "man" as "Queen" is to "woman."_

---

### 📊 **3. Get Word Vector Representation**
Every word has a **300-dimensional** vector representation.

```python
vector = model["computer"]
print(vector[:10])  # Print first 10 values
```

**Example Output:**
```
[-0.05615234  0.02392578  0.03442383 -0.05224609 ...]
```
**Explanation:**  
This vector represents the **semantic meaning of "computer"** in numerical form.

---

### 🔥 **4. Find Word Similarity Score**
Check how similar two words are using **cosine similarity**.

```python
similarity = model.similarity("king", "queen")
print(f"Similarity Score: {similarity}")
```

**Example Output:**
```
0.865
```
**Explanation:**  
The higher the score, the more semantically similar the words are.

---

### 🧩 **5. Odd One Out**
Find which word **does not** belong in a given list.

```python
odd_word = model.doesnt_match(["apple", "banana", "orange", "car"])
print(f"Odd word: {odd_word}")
```

**Example Output:**
```
'car'
```
**Explanation:**  
"Car" is not a fruit, so it is the odd one out.

---

## 📈 **Applications of Word2Vec**
1. **Search Engines** – Improve search results by understanding query intent.
2. **Chatbots** – Enhance chatbot responses using word relationships.
3. **Text Classification** – Improve sentiment analysis and spam detection.
4. **Recommendation Systems** – Suggest content based on word similarity.
5. **Machine Translation** – Understand word relationships across languages.

---

## ⚠ **Important Considerations**
- **Large Model Size**: Requires at least **4GB RAM**.
- **Vocabulary Limit**: Cannot recognize words not in the training corpus.
- **Context Sensitivity**: Doesn't understand multiple meanings of a word.

---

## 🔗 **Additional Resources**
- 📘 **Word2Vec Research Paper**: [Google Word2Vec Paper](https://arxiv.org/pdf/1301.3781.pdf)  
- 📜 **Gensim Documentation**: [Gensim Word2Vec](https://radimrehurek.com/gensim/models/keyedvectors.html)  

---

## 📜 **License**
This project uses Google's Word2Vec model under the **Google Research License**.

---

### 🚀 **Start Exploring Word2Vec Today!**
Try experimenting with different words and tasks using this model to deepen your understanding of word embeddings.

-
