import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dim", type=int, required=True, help="Embedding dimension")
parser.add_argument("--window", type=int, required=True, help="Context window size")
parser.add_argument("--neg", type=int, required=True, help="Number of negative samples")

args = parser.parse_args()

dim = args.dim
window = args.window
neg_samples = args.neg

print("\nUsing Config:")
print(f"Dimension: {dim}, Window: {window}, Negative Samples: {neg_samples}")

def load_corpus(file_path):
    corpus = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) > 2:
                corpus.append(tokens)
    return corpus

corpus = load_corpus("corpus.txt")

all_tokens = [w for sent in corpus for w in sent]
word_counts = Counter(all_tokens)

min_count = 2
vocab = [w for w, c in word_counts.items() if c >= min_count]

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

vocab_size = len(vocab)

# Convert corpus to indices
indexed_corpus = [
    [word2idx[w] for w in sent if w in word2idx]
    for sent in corpus
]

# Generating Data to train Word2Vec (Using Skip-Gram)
def generate_skipgram_data(corpus, window):
    pairs = []
    for sent in corpus:
        for i, target in enumerate(sent):
            for j in range(i - window, i + window + 1):
                if j != i and 0 <= j < len(sent):
                    pairs.append((target, sent[j]))
    return pairs

# Generating Data to train Word2Vec (Using CBOW)
def generate_cbow_data(corpus, window):
    data = []
    for sent in corpus:
        for i in range(len(sent)):
            context = []
            for j in range(i - window, i + window + 1):
                if j != i and 0 <= j < len(sent):
                    context.append(sent[j])
            if context:
                data.append((context, sent[i]))
    return data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_negative_samples(k):
    return random.choices(range(vocab_size), k=k)

# Training Using Negative Sampling
def train_skipgram(data, dim, neg_samples, epochs=20, lr=0.01):
    # Initialization
    W = np.random.randn(vocab_size, dim)
    W_prime = np.random.randn(vocab_size, dim)

    for epoch in range(epochs):
        loss = 0

        for target, context in data:
            v = W[target]
            u = W_prime[context]

            # Positive
            score = sigmoid(np.dot(v, u))
            loss += -np.log(score + 1e-9)

            grad = score - 1

            W[target] -= lr * grad * u
            W_prime[context] -= lr * grad * v

            # Negative samples
            negatives = get_negative_samples(neg_samples)

            for neg in negatives:
                u_neg = W_prime[neg]

                score_neg = sigmoid(np.dot(v, u_neg))
                loss += -np.log(1 - score_neg + 1e-9)

                grad_neg = score_neg

                W[target] -= lr * grad_neg * u_neg
                W_prime[neg] -= lr * grad_neg * v

        if (epoch == 0 or (epoch+1)%5 == 0):
            print(f"[SkipGram] Epoch {epoch+1}, Loss: {loss:.4f}")

    return W

# Training CBOW
def train_cbow(data, dim, epochs=20, lr=0.01):
    # Initialization
    W = np.random.randn(vocab_size, dim)
    W_prime = np.random.randn(vocab_size, dim)

    for epoch in range(epochs):
        loss = 0

        for context, target in data:
            context_vecs = W[context]
            # Averaging
            v_context = np.mean(context_vecs, axis=0)

            u = W_prime[target]

            score = sigmoid(np.dot(v_context, u))
            loss += -np.log(score + 1e-9)

            grad = score - 1

            for idx in context:
                W[idx] -= lr * grad * u / len(context)

            W_prime[target] -= lr * grad * v_context

        if (epoch == 0 or (epoch+1)%5 == 0):
            print(f"[CBOW] Epoch {epoch+1}, Loss: {loss:.4f}")

    return W


def cosine_similarity(vec, matrix):
    dot = np.dot(matrix, vec)
    norm_vec = np.linalg.norm(vec)
    norm_matrix = np.linalg.norm(matrix, axis=1)

    return dot / (norm_matrix * norm_vec + 1e-9)

def get_nearest_neighbors(word, embeddings, top_k=5):
    if word not in word2idx:
        print(f"{word} not in vocab")
        return []

    idx = word2idx[word]
    vec = embeddings[idx]

    sims = cosine_similarity(vec, embeddings)

    top_indices = np.argsort(-sims)[:top_k+1]

    neighbors = []
    for i in top_indices:
        if idx2word[i] != word:
            neighbors.append((idx2word[i], sims[i]))

    return neighbors[:top_k]

sg_data = generate_skipgram_data(indexed_corpus, window)
cbow_data = generate_cbow_data(indexed_corpus, window)

sg_embeddings = train_skipgram(
    sg_data,
    dim=dim,
    neg_samples=neg_samples
)

cbow_embeddings = train_cbow(
    cbow_data,
    dim=dim
)

def analogy(a, b, c, embeddings, top_k=3):
    for w in [a, b, c]:
        if w not in word2idx:
            print(f"{w} not in vocab")
            return []

    vec = embeddings[word2idx[b]] - embeddings[word2idx[a]] + embeddings[word2idx[c]]

    sims = cosine_similarity(vec, embeddings)

    top_indices = np.argsort(-sims)[:top_k+5]

    results = []
    for i in top_indices:
        word = idx2word[i]
        if word not in [a, b, c]:
            results.append(word)

    return results[:top_k]

print("\n=== Nearest Neighbors (SkipGram) ===")

words = ["research", "student", "phd", "exam"]

for w in words:
    neighbors = get_nearest_neighbors(w, sg_embeddings)
    print(f"\n{w} ->")
    for neigh, score in neighbors:
        print(f"   {neigh:15s}  {score:.4f}")

print("\n=== Analogies ===")

print("UG : BTech :: PG :", analogy("ug", "btech", "pg", sg_embeddings))
print("student : professor :: phd :", analogy("student", "professor", "phd", sg_embeddings))
print("exam : test :: assignment :", analogy("exam", "test", "assignment", sg_embeddings))

print("\n=== Nearest Neighbors (CBOW) ===")

words = ["research", "student", "phd", "exam"]

for w in words:
    neighbors = get_nearest_neighbors(w, cbow_embeddings)
    print(f"\n{w} ->")
    for neigh, score in neighbors:
        print(f"   {neigh:15s}  {score:.4f}")

print("\n=== Analogies ===")

print("UG : BTech :: PG :", analogy("ug", "btech", "pg", cbow_embeddings))
print("student : professor :: phd :", analogy("student", "professor", "phd", cbow_embeddings))
print("exam : test :: assignment :", analogy("exam", "test", "assignment", cbow_embeddings))

# TASK-4

categories = {
    "Programs": ["btech", "mtech", "phd", "degree", "dual", "program", "programs"],
    "Academics": ["course", "courses", "semester", "academic", "credits", "grade", "requirements"],
    "Institute": ["institute", "department", "campus", "iit", "jodhpur"],
    "Research": ["research", "science", "engineering", "ai"]
}

colors = ["red", "blue", "green", "orange"]

plt.figure(figsize=(10,8))

all_sg_vecs = []
all_cbow_vecs = []
all_words = []
all_colors = []

# Step 1: Collect everything
for (cat, words), color in zip(categories.items(), colors):
    words = [w for w in words if w in word2idx]
    idxs = [word2idx[w] for w in words]

    sg_vecs = sg_embeddings[idxs]
    cbow_vecs = cbow_embeddings[idxs]

    all_sg_vecs.extend(sg_vecs)
    all_cbow_vecs.extend(cbow_vecs)
    all_words.extend(words)
    all_colors.extend([color] * len(words))

pca_sg = PCA(n_components=2)
all_sg_vecs_2d = pca_sg.fit_transform(all_sg_vecs)

for i, word in enumerate(all_words):
    plt.scatter(all_sg_vecs_2d[i, 0], all_sg_vecs_2d[i, 1], color=all_colors[i])
    plt.text(all_sg_vecs_2d[i, 0], all_sg_vecs_2d[i, 1], word)

plt.title("Category Clustering (SkipGram)")
plt.savefig("category_clusters_sg_numpy.png", dpi=300)
plt.show()


pca_cbow = PCA(n_components=2)
all_cbow_vecs_2d = pca_cbow.fit_transform(all_cbow_vecs)

for i, word in enumerate(all_words):
    plt.scatter(all_cbow_vecs_2d[i, 0], all_cbow_vecs_2d[i, 1], color=all_colors[i])
    plt.text(all_cbow_vecs_2d[i, 0], all_cbow_vecs_2d[i, 1], word)

plt.title("Category Clustering (CBOW)")
plt.savefig("category_clusters_cbow_numpy.png", dpi=300)
plt.show()


top_words = [w for w, _ in word_counts.most_common(50)]

idxs = [word2idx[w] for w in top_words]
vecs = sg_embeddings[idxs]

pca = PCA(n_components=2)
vecs_2d = pca.fit_transform(vecs)

plt.figure(figsize=(12,10))

for i, word in enumerate(top_words):
    plt.scatter(vecs_2d[i,0], vecs_2d[i,1])
    plt.text(vecs_2d[i,0], vecs_2d[i,1], word)

plt.title("Top Vocabulary PCA (NumPy SkipGram)")
plt.savefig("top_vocab_numpy.png", dpi=300)
plt.show()