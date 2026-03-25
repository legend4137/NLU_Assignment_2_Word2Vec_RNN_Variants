import requests
from bs4 import BeautifulSoup
import re
import os
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def extract_iitj_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove junk
    for tag in soup(["script", "style", "nav", "header", "aside"]):
        tag.decompose()

    body = soup.body
    text_parts = []

    for element in body.descendants:
        # STOP condition
        if element.name == "div" and element.has_attr("class"):
            classes = element.get("class")
            if "footer-bg" in classes:
                break

        # Collect text
        if element.name is None:  # NavigableString
            text = element.strip()
            if text:
                text_parts.append(text)

    return " ".join(text_parts)

def clean_text(text):
    # Remove template patterns like ((...)) and {{...}}
    text = re.sub(r"\(\(.*?\)\)", " ", text)
    text = re.sub(r"\{\{.*?\}\}", " ", text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text

urls = [
    "https://iitj.ac.in/main/en/introduction",
    "https://iitj.ac.in/main/en/vision-and-mission",
    "https://iitj.ac.in/main/en/history",
    "https://iitj.ac.in/main/en/campus-infrastructure",
    "https://iitj.ac.in/main/en/sustainability-policy",

    # Administration
    "https://iitj.ac.in/office-of-director/en/office-of-director",

    # Admissions
    # "https://iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology",
    "https://iitj.ac.in/bachelor-of-technology/en/eligibility",
    "https://iitj.ac.in/bachelor-of-technology/en/hostels-facilities",
    "https://iitj.ac.in/bachelor-of-technology/en/campus-life-@-iitj",
    "https://iitj.ac.in/bachelor-of-technology/en/academic-research-facilities",
    "https://iitj.ac.in/bachelor-of-technology/en/internships-placements",

    # Regulations
    "https://iitj.ac.in/office-of-academics/en/Academic-Regulations",

    # CSE Dept
    "https://iitj.ac.in/computer-science-engineering",
    "https://iitj.ac.in/computer-science-engineering/en/undergraduate-programs",
    "https://iitj.ac.in/computer-science-engineering/en/postgraduate-programs",
    "https://iitj.ac.in/computer-science-engineering/en/programs-for-working-professionals",
    "https://iitj.ac.in/computer-science-engineering/en/doctoral-programs"
]

for i, url in enumerate(urls):
    text = extract_iitj_content(url)
    text = clean_text(text)

    words = text.split()
    text = " ".join(words[1:])

    with open(f"web_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)




def normalize_academic_terms(text):
    # Normalize variants → btech
    text = re.sub(r"\b(b[\.\s\-]?tech)\b", "btech", text, flags=re.IGNORECASE)

    # Normalize → mtech
    text = re.sub(r"\b(m[\.\s\-]?tech)\b", "mtech", text, flags=re.IGNORECASE)

    # Normalize → phd
    text = re.sub(r"\b(ph[\.\s\-]?d)\b", "phd", text, flags=re.IGNORECASE)

    # Optional extras (good for IIT context)
    text = re.sub(r"\b(b[\.\s\-]?sc)\b", "bsc", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(m[\.\s\-]?sc)\b", "msc", text, flags=re.IGNORECASE)

    return text

def custom_tokenize(text):
    # Normalize important terms FIRST
    text = normalize_academic_terms(text)
    
    # Lowercase
    text = text.lower()

    # Remove template artifacts
    text = re.sub(r"\(\(.*?\)\)", " ", text)
    text = re.sub(r"\{\{.*?\}\}", " ", text)

    # Replace hyphens with space (e.g., "state-of-the-art")
    text = re.sub(r"-", " ", text)

    # Remove apostrophes (e.g., student's → student)
    text = re.sub(r"'", "", text)

    # Keep only alphabets
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split manually (NO NLTK)
    tokens = text.split(" ")

    return tokens

stop_words = {
    "the", "is", "in", "and", "to", "of", "for", "on",
    "with", "as", "by", "an", "be", "this", "that",
    "are", "at", "from", "or", "will", "if", "td", "tr", "th"
}

def preprocess(text):
    tokens = custom_tokenize(text)

    tokens = [
        w for w in tokens
        if w not in stop_words and len(w) > 1
    ]

    return tokens

folder = "."

corpus = []
all_tokens = []

for file in os.listdir(folder):
    if file.startswith("web_") and file.endswith(".txt"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            tokens = preprocess(text)

            if len(tokens) > 0:   # skip empty files
                corpus.append(tokens)
                all_tokens.extend(tokens)

num_docs = len(corpus)
num_tokens = len(all_tokens)
vocab = set(all_tokens)
vocab_size = len(vocab)

print("Number of Documents:", num_docs)
print("Total Tokens:", num_tokens)
print("Vocabulary Size:", vocab_size)

word_freq = Counter(all_tokens)

wc = WordCloud(
    width=1000,
    height=500,
    background_color='white'
).generate_from_frequencies(word_freq)

# Save the processed corpus (one document per line)
with open("corpus.txt", "w", encoding="utf-8") as cf:
    for doc_tokens in corpus:
        cf.write(" ".join(doc_tokens) + "\n")

# Generate and save the word cloud image
plt.figure(figsize=(12,6))
plt.imshow(wc)
plt.axis("off")
plt.title("Word Cloud of IIT Jodhpur Corpus")
plt.savefig("wordcloud.png", bbox_inches="tight", dpi=300)
plt.show()