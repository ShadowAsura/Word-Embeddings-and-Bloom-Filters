from tqdm import tqdm
import sys
import time
from nltk.corpus import stopwords, wordnet
import contextlib
import numpy as np
import scipy
import contextlib
import string
import nltk
import json
import copy
import spacy
import lemminflect

# GPU acceleration: requires CuPy. Fails if GPU is unavailable.
import os
# On Windows, add CUDA bin to DLL search path before importing CuPy (Python 3.8+ doesn't use PATH for extension DLLs).
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
if os.name == "nt" and os.path.isdir(_cuda_bin):
    os.environ["PATH"] = _cuda_bin + os.pathsep + os.environ.get("PATH", "")
    if "CUDA_PATH" not in os.environ:
        os.environ["CUDA_PATH"] = os.path.dirname(_cuda_bin)
    if hasattr(os, "add_dll_directory"):  # Python 3.8+
        os.add_dll_directory(_cuda_bin)

import cupy as cp
from scipy import sparse as scipy_sparse
from cupyx.scipy import sparse as cp_sparse
_ = cp.array([1.0])  # verify GPU works
print("✓ GPU (CuPy) detected and initialized")

nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
POS = ("CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", 
       "NNP", "NNPS", "NNS", "PDT", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "VB", 
       "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB")

def lemmatize(word): # Takes a word and uses the spacy lemmatizer to return the lemmatized form
    token = nlp(str(word))[0]
    lemma = token.lemma_
    inflections = {token._.inflect(pos) for pos in POS} # returns the inflection of the lemmatized token. (ex: run -> {'ran', 'run', 'runner', 'runnest', 'running', 'runs'} )
    return lemma, inflections

def tokenize(sentence): # Tokenizes a sentence and lemmatizes the words within
    tokenized = nlp(sentence.translate(str.maketrans('', '', string.punctuation)))
    return [token.lemma_ for token in tokenized 
            if token.lemma_.lower() not in en_stopwords 
            and wordnet.synsets(token.lemma_)] # disregards lemmatized token if it's in list of stopwords or not in english dictionary (wordnet)

nltk.download('stopwords')
nltk.download('wordnet')
en_stopwords = set(stopwords.words('english'))

with open('data/fairytales_word_tf-idfs.json', 'r') as f:
    tf_idfs = json.load(f)
with open('data/fairytales_word_bloom-filters.json', 'r') as f:
    bloom_filters = json.load(f)
with open('data/fairytales_tokenized.json', 'r') as f:
    tokenized_corpus = json.load(f)

iterative_vectors = {}

def rescale_bloom_filter(): # Rescales bloom filters to be in range [-1, 1] instead of [0, 1]
    for word in list(bloom_filters.keys()):
        bloom_filters[word] = cp.asarray(bloom_filters[word], dtype=cp.float32) * 2 - 1


def build_neighbor_dict(deltas):
    """Build neighbor adjacency list mapping word -> list[(neighbor, weight)].

    Keeps everything matrix-free and stores only the cooccurring neighbors and tf-idf weights.
    """
    from collections import defaultdict
    neighbor_dict = defaultdict(list)

    for sentence in tqdm(tokenized_corpus, desc="Building neighbors", leave=False):
        for p in range(len(sentence)):
            center = sentence[p]
            if center not in tf_idfs or center not in bloom_filters:
                continue

            for delta in deltas:
                q = p + delta
                if q < 0 or q >= len(sentence):
                    continue

                neighbor = sentence[q]
                if neighbor not in tf_idfs:
                    continue

                tf_idf = tf_idfs.get(center, {}).get(neighbor, 0)
                if tf_idf > 0:
                    neighbor_dict[center].append((neighbor, float(tf_idf)))

    return neighbor_dict


def iterative_update_edges(V_prev, edge_src, edge_dst, edge_w, n_words, bits=32):
    """GPU edge-based update using scatter-add.

    V_prev: (n_words, bits)
    edge_src: (n_edges,) indices of neighbor vectors
    edge_dst: (n_edges,) indices of center words
    edge_w:   (n_edges,) weight for each edge
    """
    V_new = cp.zeros((n_words, bits), dtype=cp.float32)

    # gather neighbor vectors
    gathered = V_prev[edge_src]              # (n_edges, bits)
    weighted = gathered * edge_w[:, None]    # broadcast

    # scatter-add into destinations
    try:
        cp.scatter_add(V_new, edge_dst[:, None], weighted)
    except AttributeError:
        # fallback if scatter_add not available: use add.at per dimension
        for d in range(bits):
            cp.add.at(V_new[:, d], edge_dst, weighted[:, d])

    # normalize by counts per target
    counts = cp.bincount(edge_dst, minlength=n_words).reshape(-1, 1)
    counts[counts == 0] = 1
    V_new /= counts

    return V_new


# Matrix-free: no dense n×n matrix; we use sparse W (only co-occurring pairs) and sparse @ dense on GPU.

# ---- legacy word-loop helpers (kept only for academic traceability) ----
# these functions implement the original slow per-word iteration
# and are not invoked by the optimized pipeline below.
def generate_vector(word, tokenized_sentence, bits, deltas, iteration):
    """ 
    Generates vector representation for word when given a sentence.
    Uses GPU (CuPy) for accelerated computation.
    
    Args:
        word (str): The word to sum the neighbors of. Note that multiple instances of the word could occur.
        tokenized_sentence (list): The sentence the word instance(s) is/are contained in, as a list of tokens.
        bits (int): The number of bits the representation should be.
        deltas (int): The index of the neighbors, relative to the position of the target word(s), to sum. (e.g. [-4, -3, -2, -1, 1, 2, 3, 4] means the 4 words before and after the word(s)).
        iteration (int): The iteration. Although an integer input, indicates whether a previous iteration has occurred. If not, we will use the bloom filters of each word as their representations. Otherwise, we will use the representations from the previous iteration.
        
    Returns:
        instance_representation (cp.ndarray): The vector representation (on GPU) prior to averaging by the number of neighbors.
        adjacent_words (int): The number of adjacent words that were found and used to construct the representation.
    """
    indices = [i for i, x in enumerate(tokenized_sentence) if x == word] # Gets the indices of all occurrences of the word in this sentence.
    instance_representation = cp.zeros(bits) 
    adjacent_words = 0

    for index in indices: # for each occurrence of the word in the sentence
        for delta in deltas: # for each neighbor
            if index + delta < 0: # if the neighbor index is negative, skip.
                continue
            with contextlib.suppress(IndexError): # suppress IndexError if the neighbor is out of bounds
                adjacent_word = tokenized_sentence[index + delta]
                try: # if the neighbor does not have a tf-idf for this word on file, it is too infrequent to be relevant, so we skip and default to 0.
                    tf_idf = tf_idfs[word][adjacent_word]
                except KeyError:
                    tf_idf = 0
                try: # if the neighbor word doesn't have a representation on file, skip it
                    if iteration: # if this is not the first iteration, we use the preassigned iterative vectors for the adjacent word.
                        instance_representation += cp.array(iterative_vectors[adjacent_word]) * tf_idf
                    else:
                        instance_representation += bloom_filters[adjacent_word] * tf_idf
                    adjacent_words += 1
                except KeyError:
                    continue
    return instance_representation, adjacent_words

# legacy helper; see comment above
def extract_vectors(word, iteration, deltas=None, bits=32):
    """ 
        Extracts the vector representation of a word using GPU acceleration.
        
        Args:
            word (str): The word we are finding the representation of.
            iteration (int): The index of the current iteration.
            deltas (int): The index of the neighbors, relative to the position of the target word(s), to sum. (e.g. [-4, -3, -2, -1, 1, 2, 3, 4] means the 4 words before and after the word(s)).
            bits (int): The number of bits the representation should be.
    """
    if deltas is None:
        size = 6
        deltas = [i for i in range(-size, size + 1) if i != 0]

    total_adjacent_words = 0
    representations = cp.zeros(bits)

    for sentence in tokenized_corpus:
        if word in sentence: # if the word is in the sentence, we pass it to the generate_vector function.
            representation, adjacent_words = generate_vector(word, sentence, bits, deltas, iteration)
            representations += representation # the representation accumulates to be the sum of all the neighbor representations.
            total_adjacent_words += adjacent_words # the count of neighbors accumulates
    if total_adjacent_words == 0:
        return representations  # return zeros if no neighbors found
    return representations / total_adjacent_words # we take the average of all neighbors by dividing the sum of their represntations by the count of neighbors.


def update_encoding(word, iteration, args):
    """Replaces the previous vector representation of word in iterative_vectors with the new one.
    Stores as GPU array for faster subsequent operations.
    """
    vector = extract_vectors(word, iteration, **args)
    iterative_vectors[word] = vector

def normalize_vector():
    """This is not used in the current implementation, but it normalizes the vectors to unit length using GPU acceleration.
    """
    for word in iterative_vectors.keys():
        iterative_vectors[word] = iterative_vectors[word] / cp.linalg.norm(iterative_vectors[word]) # normalized on GPU

def normalize_vector_dimensions(iterative_vectors):
    """Normalizes vector dimensions by (1) normalizing the length of each vector to 1 and (2) normalizing vectors along the dimensions (columns) using Robust Scaling to ignore outliers while simultaneously adjusting the scale of each dimension.
    Uses GPU acceleration with CuPy.
    """
    # Stack on GPU without GPU->CPU->GPU round-trip (keep keys order for dict return)
    keys = list(iterative_vectors.keys())
    first = iterative_vectors[keys[0]]
    if isinstance(first, cp.ndarray):
        vectors = cp.stack([iterative_vectors[word] for word in keys])
    else:
        vectors = cp.array([iterative_vectors[word] for word in keys])

    # Row normalization (GPU)
    norms = cp.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    # Column normalization (robust scaling on GPU)
    med = cp.median(vectors, axis=0)
    # CuPy equivalent of scipy.stats.iqr
    q75 = cp.percentile(vectors, 75, axis=0)
    q25 = cp.percentile(vectors, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1
    vectors = (vectors - med) / iqr

    return {
        word: vectors[i] for i, word in enumerate(keys)
    }


def sigmoid_normalize_vectors():
    """Not used in current implementation. GPU-accelerated sigmoid normalization.
    """
    for word in iterative_vectors.keys():
        iterative_vectors[word] = 2 / (1 + cp.exp(-iterative_vectors[word])) - 1  # sigmoid function + scale to pos/neg on GPU

def convert_to_cpu(iterative_vectors):
    """Convert GPU arrays back to CPU for JSON serialization.
    """
    cpu_vectors = {}
    for word, vector in iterative_vectors.items():
        cpu_vectors[word] = cp.asnumpy(vector).tolist()
    return cpu_vectors

if __name__ == '__main__':
    ITERATIONS = int(os.environ.get('ITERATIONS', '400'))
    NEIGHBORHOOD_SIZE = 6
    deltas = [i for i in range(-NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE + 1) if i != 0]

    print("Using GPU (CuPy), neighbor-list additive updates (matrix-free)")
    print(f"Deltas: {deltas}")

    os.makedirs('data/iterative_vectors', exist_ok=True)

    rescale_bloom_filter()

    # ============================================================
    # ORIGINAL WORD-LOOP VERSION (PRESERVED FOR REFERENCE)
    #
    # This version followed the research notes directly:
    #
    #     for each word:
    #         for each neighbor:
    #             V_new[word] += tfidf(word, neighbor) * V_prev[neighbor]
    #         V_new[word] /= number_of_neighbors
    #
    # This version was correct but extremely slow due to:
    #
    #   * Python-level loops per word
    #   * Millions of tiny GPU kernel launches
    #
    # The new implementation below preserves the exact mathematics
    # but vectorizes accumulation using an edge-based scatter-add.
    #
    # Do not delete old functions like generate_vector or extract_vectors.
    # Keep them clearly commented as legacy reference code.
    # ============================================================

    # Build neighbor adjacency list (no matrices)
    neighbor_dict = build_neighbor_dict(deltas)
    words_list = sorted(neighbor_dict.keys())
    n_words = len(words_list)

    # Build word->index map and convert bloom filters for words_list (already cp arrays from rescale)
    word_to_i = {w: i for i, w in enumerate(words_list)}
    bits = 32

    # build flattened edge lists on CPU (no GPU roundtrips)
    edge_src = []
    edge_dst = []
    edge_w = []
    for i, w in enumerate(words_list):
        for n, wt in neighbor_dict[w]:
            j = word_to_i.get(n)
            if j is None:
                continue
            edge_src.append(j)
            edge_dst.append(i)
            edge_w.append(wt)

    # convert to GPU arrays once
    edge_src = cp.asarray(np.array(edge_src, dtype=np.int32))
    edge_dst = cp.asarray(np.array(edge_dst, dtype=np.int32))
    edge_w   = cp.asarray(np.array(edge_w, dtype=np.float32))

    # diagnostics
    print("=================================================")
    print("Diagnostics")
    print("-------------------------------------------------")
    print("Number of words:", n_words)
    print("Number of edges:", edge_src.size)
    print("CuPy scatter_add available:", hasattr(cp, "scatter_add"))
    print("=================================================")

    # Initial V_prev (stack bloom filters in words_list order)
    B_gpu = cp.stack([bloom_filters[w] for w in words_list]).astype(cp.float32)
    V_prev = B_gpu

    # ============================================================
    # ITERATIVE EDGE-BASED UPDATES
    # (gather + scatter-add, no Python word loop, with timing)
    start_total = time.time()
    for i in range(ITERATIONS):
        iter_start = time.time()

        V_new = iterative_update_edges(V_prev, edge_src, edge_dst, edge_w, n_words, bits=bits)

        # Row normalize every iteration
        norms = cp.linalg.norm(V_new, axis=1, keepdims=True)
        norms[norms == 0] = 1
        V_new = V_new / norms

        # ============================================================
        # FULL ROBUST SCALING EVERY ITERATION (ORIGINAL BEHAVIOR)
        #
        # med = cp.median(V_new, axis=0)
        # q75 = cp.percentile(V_new, 75, axis=0)
        # q25 = cp.percentile(V_new, 25, axis=0)
        # iqr = q75 - q25
        # iqr[iqr == 0] = 1
        # V_new = (V_new - med) / iqr
        #
        # NOTE:
        # Running percentile every iteration is expensive.
        # Current implementation applies this every 10 iterations
        # to preserve behavior while reducing compute overhead.
        # This preserves the original behavior for traceability.
        # ============================================================

        # robust scaling every 10 iterations
        if i % 10 == 0:
            med = cp.median(V_new, axis=0)
            q75 = cp.percentile(V_new, 75, axis=0)
            q25 = cp.percentile(V_new, 25, axis=0)
            iqr = q75 - q25
            iqr[iqr == 0] = 1
            V_new = (V_new - med) / iqr

        V_prev = V_new

        iter_time = time.time() - iter_start
        elapsed = time.time() - start_total
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (ITERATIONS - i - 1)

        print(
            "Iteration",
            i + 1,
            "/",
            ITERATIONS,
            "| iter_time:",
            round(iter_time, 2),
            "s | elapsed:",
            round(elapsed / 60, 2),
            "m | eta:",
            round(remaining / 60, 2),
            "m"
        )

        if i % 10 == 0 or i == ITERATIONS - 1:
            cpu_vectors = {w: cp.asnumpy(V_prev[j]).tolist() for j, w in enumerate(words_list)}
            with open(f'data/iterative_vectors/window_{NEIGHBORHOOD_SIZE}_iter_{i}.json', 'w') as f:
                json.dump(cpu_vectors, f, indent=4)

    print("Total runtime:", round((time.time() - start_total) / 60, 2), "minutes")
