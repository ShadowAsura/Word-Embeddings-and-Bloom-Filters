"""
Regenerate the three fairytales data files from raw fairy_tales/*.txt:
  data/fairytales_tokenized.json
  data/fairytales_word_tf-idfs.json
  data/fairytales_word_bloom-filters.json
"""
import contextlib
import json
import math
import string
from pathlib import Path

import mmh3
import nltk
import spacy
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
en_stopwords = set(stopwords.words('english'))

# ---------------------------------------------------------------------------
# Step 1 – Tokenize
# ---------------------------------------------------------------------------
def tokenize_sentence(sentence):
    cleaned = sentence.translate(str.maketrans('', '', string.punctuation))
    doc = nlp(cleaned)
    return [
        token.lemma_
        for token in doc
        if token.lemma_.lower() not in en_stopwords and wordnet.synsets(token.lemma_)
    ]

FAIRY_TALES_DIR = Path('fairy_tales')
tokenized_path = Path('data/fairytales_tokenized.json')

if tokenized_path.exists() and tokenized_path.stat().st_size > 1000:
    print("Loading existing tokenized corpus…")
    with open(tokenized_path) as f:
        all_sentences = json.load(f)
    print(f"  {len(all_sentences)} sentences loaded")
else:
    print("Tokenizing fairy tales…")
    all_sentences = []
    for i in range(1, 1731):
        path = FAIRY_TALES_DIR / f'{i}.txt'
        with contextlib.suppress(FileNotFoundError):
            raw = path.read_bytes()
            try:
                text = raw.decode('cp1252').lower()
            except UnicodeDecodeError:
                text = raw.decode('utf8').lower()
            for sentence in sent_tokenize(text):
                tokens = tokenize_sentence(sentence)
                if tokens:
                    all_sentences.append(tokens)
        if i % 200 == 0:
            print(f"  {i}/1730 files — {len(all_sentences)} sentences")

    with open(tokenized_path, 'w') as f:
        json.dump(all_sentences, f)
    print(f"Saved {len(all_sentences)} sentences → {tokenized_path}")

# ---------------------------------------------------------------------------
# Step 2 – Compute co-occurrence neighbor frequencies (window ±10)
# ---------------------------------------------------------------------------
print("Computing co-occurrence frequencies (window ±10)…")
WINDOW = 10
neighbor_frequencies = {}   # word -> {neighbor: count}
neighbor_total = {}         # word -> total times it appears as any neighbor

words_in_corpus = sum(len(s) for s in all_sentences)
print(f"  Total tokens in corpus: {words_in_corpus}")

for sent_idx, sentence in enumerate(all_sentences):
    if sent_idx % 20000 == 0:
        print(f"  sentence {sent_idx}/{len(all_sentences)}")
    n = len(sentence)
    for p in range(n):
        center = sentence[p]
        if center not in neighbor_frequencies:
            neighbor_frequencies[center] = {}
        for delta in range(-WINDOW, WINDOW + 1):
            if delta == 0:
                continue
            q = p + delta
            if 0 <= q < n:
                nbr = sentence[q]
                neighbor_frequencies[center][nbr] = neighbor_frequencies[center].get(nbr, 0) + 1
                neighbor_total[nbr] = neighbor_total.get(nbr, 0) + 1

print(f"  Vocabulary size: {len(neighbor_frequencies)}")

# n_neighbors[word] = sum of all co-occurrence counts for word (denominator in TF)
n_neighbors_word = {w: sum(d.values()) for w, d in neighbor_frequencies.items()}

# ---------------------------------------------------------------------------
# Step 3 – TF-IDF (per generate_tf-idfs_words.ipynb cell 5 formula)
#   tf_idfs[word][neighbor] = log( freq(w,n)/n_neighbors[w]  *  log(corpus/n_neighbors[n]) )
# then shift by –minimum so all values ≥ 0
# ---------------------------------------------------------------------------
print("Computing TF-IDFs…")
tf_idfs = {}
for word, neighbors in neighbor_frequencies.items():
    denom_w = n_neighbors_word[word]
    tf_idfs[word] = {}
    for nbr, cnt in neighbors.items():
        denom_nbr = neighbor_total.get(nbr, 1)
        idf_val = math.log(words_in_corpus / denom_nbr)
        if idf_val <= 0:
            continue
        val = math.log(cnt / denom_w * idf_val)
        tf_idfs[word][nbr] = val

# Sort each word's neighbors by TF-IDF descending
for word in tf_idfs:
    tf_idfs[word] = dict(sorted(tf_idfs[word].items(), key=lambda x: x[1], reverse=True))

# Shift so minimum = 0 (matches notebook cell 8)
all_vals = [v for d in tf_idfs.values() for v in d.values()]
minimum = min(all_vals)
print(f"  TF-IDF minimum before shift: {minimum:.4f}")
for word in tf_idfs:
    for nbr in tf_idfs[word]:
        tf_idfs[word][nbr] -= minimum

tfidf_path = Path('data/fairytales_word_tf-idfs.json')
with open(tfidf_path, 'w') as f:
    json.dump(tf_idfs, f)
print(f"Saved {len(tf_idfs)} words → {tfidf_path}")

# ---------------------------------------------------------------------------
# Step 4 – Bloom filters (per generate_bloom_filters.ipynb)
# ---------------------------------------------------------------------------
print("Computing Bloom filters…")

def bloom_filter(word, bits=32):
    array = [0] * bits
    encoding = mmh3.hash(word)
    i = 31
    while encoding > 0:
        remainder = encoding % 2
        array[i] = remainder
        i -= 1
        encoding //= 2
    return array

bloom_filters = {}
for sentence in all_sentences:
    for word in sentence:
        if word not in bloom_filters:
            bloom_filters[word] = bloom_filter(word)

bloom_path = Path('data/fairytales_word_bloom-filters.json')
with open(bloom_path, 'w') as f:
    json.dump(bloom_filters, f)
print(f"Saved {len(bloom_filters)} words → {bloom_path}")

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
print("\n--- Verification ---")
for p in [tokenized_path, tfidf_path, bloom_path]:
    size_mb = p.stat().st_size / 1e6
    with open(p) as f:
        first_char = f.read(1)
    print(f"{p}: {size_mb:.2f} MB  first_char={first_char!r}")

print("Done.")
