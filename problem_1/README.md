# Word2Vec from Scratch: IIT Jodhpur Web Corpus

This repository contains a complete pipeline for scraping domain-specific text, creating a custom academic corpus, and training Word2Vec word embeddings (Skip-Gram and CBOW) entirely from scratch using NumPy.

## Project Structure

The workflow is divided into two primary scripts:

### 1. `create_corpus.py` (Data Collection & Preprocessing)
* **Web Scraping:** Uses `BeautifulSoup` to extract raw text from various IIT Jodhpur web pages (Administration, Academics, CSE Dept, etc.).
* **Text Cleaning:** Removes HTML tags, scripts, and layout artifacts.
* **Custom Tokenization:** Normalizes academic terms (e.g., standardizing formatting for BTech, MTech, PhD), removes punctuation, and filters out custom stop words **without** relying on heavy external NLP libraries like NLTK.
* **Outputs:** * `corpus.txt`: The final preprocessed dataset (one document per line) used for training.
  * `wordcloud.png`: A visual representation of the most frequent terms in the corpus.
  * `web_*.txt`: Intermediate raw text files for each scraped URL.

### 2. `train_word2vec.py` (Model Training & Evaluation)
* **Architectures:** Implements both **Skip-Gram** (optimized with Negative Sampling) and **Continuous Bag of Words (CBOW)** from scratch.
* **Evaluation:** Calculates cosine similarity to find the nearest neighbors for specific academic terms and evaluates word analogies (e.g., `ug : btech :: pg : ?`).
* **Visualization:** Applies Principal Component Analysis (PCA) to project the high-dimensional word vectors into a 2D space for visual clustering.
* **Outputs:**
  * `category_clusters_sg_numpy.png`: PCA plot of clustered academic categories using Skip-Gram.
  * `category_clusters_cbow_numpy.png`: PCA plot of clustered academic categories using CBOW.
  * `top_vocab_numpy.png`: PCA plot of the top 50 most frequent words in the corpus.

---

## Requirements

Install the necessary Python dependencies before running the scripts:

```bash
pip install requests beautifulsoup4 wordcloud matplotlib numpy scikit-learn
```

---

## Usage

### Step 1: Generate the Corpus
Run the web scraper and preprocessor. This must be executed first to generate the `corpus.txt` file.
```bash
python create_corpus.py
```

### Step 2: Train the Embeddings
Run the Word2Vec training script. This script uses command-line arguments to configure the model's hyperparameters.

**Example Run:**
```bash
python train_word2vec.py --dim 50 --window 2 --neg 5
```

**Command-Line Arguments:**

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--dim` | `int` | **Required.** The dimensionality of the word embedding vectors. |
| `--window` | `int` | **Required.** The context window size (number of words to look at before and after the target word). |
| `--neg` | `int` | **Required.** The number of negative samples to draw per positive pair in the Skip-Gram model. |

---

## Expected Output
During execution, `train_word2vec.py` will print the training loss at regular intervals. Once training is complete, it will output:
1. The top 5 nearest neighbors for test words (e.g., "research", "student", "phd", "exam").
2. The results of vector algebra analogies.
3. Automatically generated Matplotlib windows showing the 2D PCA projections of your custom embeddings.