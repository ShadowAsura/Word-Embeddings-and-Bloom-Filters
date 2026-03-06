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

eps = 1e-8 # Small epsilon for numerical stability

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

print(f"Size of tf_idfs.keys(): {len(tf_idfs.keys())}")

iterative_vectors = {}

def rescale_bloom_filter(): # Rescales bloom filters to be in range [-1, 1] instead of [0, 1]
    for word in list(bloom_filters.keys()):
        bloom_filters[word] = cp.asarray(bloom_filters[word], dtype=cp.float32) * 2 - 1


def build_neighbor_dict(deltas, vocabulary_filter):
    """Build neighbor adjacency list mapping word -> list[(neighbor, weight)].

    Keeps everything matrix-free and stores only the cooccurring neighbors and tf-idf weights.
    Only includes words present in the vocabulary_filter.
    """
    from collections import defaultdict
    neighbor_dict = defaultdict(list)

    for sentence in tqdm(tokenized_corpus, desc="Building neighbors", leave=False):
        for p in range(len(sentence)):
            center = sentence[p]
            if center not in vocabulary_filter: # Filter by fixed vocabulary
                continue

            for delta in deltas:
                q = p + delta
                if q < 0 or q >= len(sentence):
                    continue

                neighbor = sentence[q]
                if neighbor not in vocabulary_filter: # Filter by fixed vocabulary
                    continue

                tf_idf = tf_idfs.get(center, {}).get(neighbor, 0.0) # Ensure float type
                neighbor_dict[center].append((neighbor, tf_idf))

    return neighbor_dict


def safe_row_normalize(X, eps=1e-8):
    """Safely normalize rows to unit length, handling NaN/Inf."""

    norms = cp.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return X / norms

BATCH_SIZE = 2**16 # Define a global or configurable batch size for GPU operations

def iterative_update_edges(
    V_prev,
    edge_src,
    edge_dst,
    edge_w, # raw TF-IDF weights, not normalized
    n_words,
    bits,
    eps=1e-8
):
    """GPU edge-based update: accumulates weighted neighbor vectors using batching.
    This function computes the numerator part of the update rule.

    V_prev: (n_words, bits) - current vectors (unit-normalized)
    edge_src, edge_dst: (n_edges,) - edge indices
    edge_w: (n_edges,) - raw TF-IDF weights (numerator only)
    """
    V_diff = cp.zeros((n_words, bits), dtype=cp.float32)
    n_edges = edge_src.shape[0]

    for i in tqdm(range(0, n_edges, BATCH_SIZE), desc="Batching edge updates", leave=False):
        batch_slice = slice(i, min(i + BATCH_SIZE, n_edges))

        gathered_batch = V_prev[edge_src[batch_slice]]
        weighted_batch = gathered_batch * edge_w[batch_slice, None] # Apply raw TF-IDF weights

        try:
            cp.scatter_add(V_diff, edge_dst[batch_slice, None], weighted_batch)
        except AttributeError: # Fallback for older CuPy versions if scatter_add doesn't work with None
            for d in range(bits):
                cp.add.at(V_diff[:, d], edge_dst[batch_slice], weighted_batch[:, d])
            
    return V_diff # Return raw accumulated numerator


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

def precompute_denominator_counts(
    tokenized_corpus,
    words_list, # This is list(tf_idfs.keys())
    initial_vector_keys, # set of keys that have initial vectors (e.g., from 0.json or bloom_filters)
    neighborhood_size # equivalent to x in CPU deltas
):
    """
    Precomputes the denominator counts for each word, matching CPU logic for
    iteration 0 and iteration >= 1.
    """
    n_words = len(words_list)
    word_to_idx = {word: i for i, word in enumerate(words_list)}

    # Initialize CPU-like deltas
    deltas = [i for i in range(-neighborhood_size, neighborhood_size + 1) if i != 0]

    # Denominator for iteration 0: neighbor must exist in initial_vector_keys
    counts_iter0 = np.zeros(n_words, dtype=np.int32)
    # Denominator for iteration >= 1: neighbor must exist in current embedding vocab (words_list)
    counts_iter_n = np.zeros(n_words, dtype=np.int32)

    for sentence_idx, tokenized_sentence in enumerate(tqdm(tokenized_corpus, desc="Precomputing denominators", dynamic_ncols=True)):
        for word_idx_in_sentence, current_word in enumerate(tokenized_sentence):
            if current_word not in word_to_idx:
                continue # Only count for words in our vocabulary

            current_word_global_idx = word_to_idx[current_word]

            for delta in deltas:
                neighbor_idx_in_sentence = word_idx_in_sentence + delta

                # In-bounds check
                if 0 <= neighbor_idx_in_sentence < len(tokenized_sentence):
                    neighbor_word = tokenized_sentence[neighbor_idx_in_sentence]

                    # Condition for iter 0: neighbor must be in initial_vector_keys
                    if neighbor_word in initial_vector_keys:
                        counts_iter0[current_word_global_idx] += 1

                    # Condition for iter >= 1: neighbor must be in words_list (embedding vocab)
                    # Note: word_to_idx implicitly checks for existence in words_list
                    if neighbor_word in word_to_idx:
                        counts_iter_n[current_word_global_idx] += 1
    
    # Convert to CuPy arrays
    counts_iter0_cp = cp.array(counts_iter0, dtype=cp.float32)
    counts_iter_n_cp = cp.array(counts_iter_n, dtype=cp.float32)
    
    # Handle division by zero: if a word has no valid neighbors, its count will be 0.
    # We should ensure the denominator is at least 1 to avoid NaN/Inf.
    # This matches the spirit of the CPU code where `total_adjacent_words` could be 0, leading to issues.
    # For robust GPU computation, we should replace 0 with 1 to avoid NaNs.
    counts_iter0_cp[counts_iter0_cp == 0] = 1
    counts_iter_n_cp[counts_iter_n_cp == 0] = 1

    return counts_iter0_cp, counts_iter_n_cp


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
    NUM_ITERATIONS_TO_RUN = int(os.environ.get('ITERATIONS', '400'))
    NEIGHBORHOOD_SIZE = 6
    deltas = [i for i in range(-NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE + 1) if i != 0]

    print("Using GPU (CuPy), neighbor-list additive updates (matrix-free)")
    print(f"Deltas: {deltas}")

    os.makedirs('data/iterative_vectors', exist_ok=True)

    rescale_bloom_filter()

    # Define the fixed vocabulary for all iterations based on tf_idfs keys
    # This ensures strict vocabulary equivalence with the CPU ground truth.
    fixed_vocab_words = sorted(list(tf_idfs.keys()))
    n_words_fixed_vocab = len(fixed_vocab_words)
    word_to_idx_fixed_vocab = {w: i for i, w in enumerate(fixed_vocab_words)}

    # Determine 'bits' from the first word in fixed_vocab_words
    bits = 200 # Default if no words in bloom_filters
    if fixed_vocab_words:
        bits = bloom_filters[fixed_vocab_words[0]].shape[0]







    # Precompute denominators based on CPU logic (corpus scanning)
    bloom_filters_keys_set = set(bloom_filters.keys()) # This is effectively the same as fixed_vocab_words_set
    counts_iter0_cp, counts_iter_n_cp = precompute_denominator_counts(
        tokenized_corpus=tokenized_corpus,
        words_list=fixed_vocab_words,
        initial_vector_keys=bloom_filters_keys_set, # Use bloom_filters_keys for iter0 counts as per CPU logic
        neighborhood_size=NEIGHBORHOOD_SIZE
    )

    # ============================================================
    # ITERATIVE EDGE-BASED UPDATES
    # (gather + scatter-add, with timing)
    # This section implements the mathematically equivalent GPU version
    # of the CPU code, following all user constraints.
    # ============================================================
    start_total = time.time()

    # Load 0.json as the initial V_prev to match CPU ground truth
    # This will override the `fixed_vocab_words` and `bits` determined earlier
    # to ensure strict alignment with the CPU's iteration 0 output.
    initial_vectors_path = os.path.join('data/iterative_vectors', '0.json')
    with open(initial_vectors_path, 'r') as f:
        cpu_initial_vectors = json.load(f)

    # Update fixed_vocab_words and bits from the loaded 0.json
    cpu_initial_words = list(cpu_initial_vectors.keys())
    if sorted(cpu_initial_words) != sorted(fixed_vocab_words):
        raise ValueError("Vocabulary from 0.json does not match tf_idfs vocabulary.")
    
    if fixed_vocab_words:
        bits = len(cpu_initial_vectors[fixed_vocab_words[0]])
    else:
        raise ValueError("0.json is empty or ill-formatted, cannot determine vector dimensions.")

    # Rebuild word_to_idx_fixed_vocab based on the loaded 0.json keys (which should be identical to tf_idfs keys)
    word_to_idx_fixed_vocab = {w: i for i, w in enumerate(fixed_vocab_words)}

    print(f"Size of fixed_vocab_words (from tf_idfs.keys()): {len(fixed_vocab_words)}")
    print(f"Size of bloom_filters_keys_set: {len(bloom_filters_keys_set)}")
    common_words_count = len(set(fixed_vocab_words).intersection(bloom_filters_keys_set))
    print(f"Number of common words between 0.json vocab and bloom filter vocab: {common_words_count}")
    
    if len(fixed_vocab_words) != common_words_count:
        print("WARNING: Vocabulary mismatch between 0.json and bloom filters. This might cause issues.")

    # Initialize V_prev from the loaded 0.json vectors
    V_prev = cp.stack([cp.asarray(cpu_initial_vectors[w]) for w in fixed_vocab_words]).astype(cp.float32)

    # Rebuild neighbor adjacency list and edge lists with the updated fixed_vocab_words
    # This is necessary because fixed_vocab_words might have changed from the initial definition
    print("Rebuilding neighbor dictionary and edge lists with 0.json vocabulary...")
    neighbor_dict = build_neighbor_dict(deltas, set(fixed_vocab_words))

    # build flattened edge lists on CPU (no GPU roundtrips)
    edge_src = []
    edge_dst = []
    edge_w = []
    for i, w in enumerate(fixed_vocab_words): # Iterate over fixed vocabulary
        if w not in neighbor_dict: # Word might not have any valid neighbors within fixed_vocab
            continue
        for n, wt in neighbor_dict[w]:
            j = word_to_idx_fixed_vocab.get(n) # Look up in fixed vocabulary map
            if j is None: # Should not happen if neighbor_dict was built correctly
                continue
            edge_src.append(j)
            edge_dst.append(i)
            edge_w.append(wt)

    edge_src = cp.asarray(np.array(edge_src, dtype=np.int32))
    edge_dst = cp.asarray(np.array(edge_dst, dtype=np.int32))
    edge_w   = cp.asarray(np.array(edge_w, dtype=cp.float32))

    # diagnostics
    print("=================================================")
    print("Diagnostics")
    print("-------------------------------------------------")
    print("Number of words (fixed vocabulary):", n_words_fixed_vocab)
    print("Number of edges:", edge_src.size)
    print("CuPy scatter_add available:", hasattr(cp, "scatter_add"))
    print("=================================================")


    # Precompute denominators based on CPU logic (corpus scanning) with the new vocabulary
    # This block was previously a duplicate and has been removed.

    # Save iteration 0 directly from the loaded 0.json
    iterative_vectors_dict_iter0 = {fixed_vocab_words[j]: V_prev[j] for j in range(n_words_fixed_vocab)}
    cpu_vectors_iter0 = convert_to_cpu(iterative_vectors_dict_iter0)
    output_path_iter0 = os.path.join('data/iterative_vectors', f'window_{NEIGHBORHOOD_SIZE}_iter_0_v3_{bits}bit.json')
    with open(output_path_iter0, 'w') as f:
        json.dump(cpu_vectors_iter0, f, indent=4)
    print(f"Saved initial vectors (from 0.json) to {output_path_iter0}")

    # Main iteration loop for i = 1 to ITERATIONS-1 (since 0 is pre-loaded)
    for i in range(1, NUM_ITERATIONS_TO_RUN + 1):
        iter_start = time.time()

        # Determine which denominator counts to use
        # Since iteration 0 is loaded, 'i' now represents the iteration number 1-indexed
        current_denominator_counts = counts_iter_n_cp # For i >= 1, always use counts_iter_n_cp
        # Expand denominator counts to (n_words, 1) for broadcasting
        current_denominator_expanded = cp.expand_dims(current_denominator_counts, axis=1);

        numerator_vectors = iterative_update_edges(
                V_prev, edge_src, edge_dst, edge_w, n_words_fixed_vocab, bits=bits
            )

        V_updated_raw = numerator_vectors / current_denominator_expanded
        
        # Apply normalization: (1) Row L2 normalization, then (2) Robust column scaling
        # This order matches the CPU version's normalize_vector_dimensions function.
        # (1) Row L2 normalization
        V_normalized_rows = safe_row_normalize(V_updated_raw)

        # (2) Robust column scaling
        med = cp.median(V_normalized_rows, axis=0)
        q75 = cp.percentile(V_normalized_rows, 75, axis=0)
        q25 = cp.percentile(V_normalized_rows, 25, axis=0)
        iqr = q75 - q25

        # Create a mask for dimensions where IQR is not zero
        nonzero_iqr_mask_iter = (iqr != 0)

        # Initialize scaled_vectors with original V_normalized_rows
        scaled_V_next = V_normalized_rows.copy()

        # Apply scaling only where IQR is not zero
        # For dimensions where IQR is zero, scaled_V_next[:, d] remains original
        if cp.any(nonzero_iqr_mask_iter): # Only apply if there's at least one non-zero IQR dimension
            scaled_V_next[:, nonzero_iqr_mask_iter] = (V_normalized_rows[:, nonzero_iqr_mask_iter] - med[nonzero_iqr_mask_iter]) / iqr[nonzero_iqr_mask_iter]

        V_next = scaled_V_next

        # Assign to V_prev for next iteration
        V_prev = V_next
        iter_end = time.time()
        print(f"Iteration {i} took {iter_end - iter_start:.2f} seconds.")
        # Save current iteration vectors
        # Check for NaNs or Infs after computing V_next
        if cp.any(cp.isnan(V_next)) or cp.any(cp.isinf(V_next)):
            print(f"WARNING: NaN or Inf detected in V_next at iteration {i}")
            # Optionally, print more details for debugging
            nan_indices = cp.argwhere(cp.isnan(V_next))
            inf_indices = cp.argwhere(cp.isinf(V_next))
            if nan_indices.size > 0:
                print(f"NaN indices: {nan_indices[:5]}...")
            if inf_indices.size > 0:
                print(f"Inf indices: {inf_indices[:5]}...")
            
        # Convert CuPy array to Python list for JSON serialization
        # (only for saving, V_next remains a CuPy array for next iteration)
        current_iter_vectors = {fixed_vocab_words[j]: V_next[j].tolist() for j in range(n_words_fixed_vocab)}
        output_path = os.path.join('data/iterative_vectors', f'window_{NEIGHBORHOOD_SIZE}_iter_{i}_v3_{bits}bit.json')
        with open(output_path, 'w') as f:
            json.dump(current_iter_vectors, f, indent=4)
        print(f"Saved iteration {i} vectors to {output_path}")
    
    end_total = time.time()
    print(f"Total time for {NUM_ITERATIONS_TO_RUN} iterations: {end_total - start_total:.2f} seconds.")