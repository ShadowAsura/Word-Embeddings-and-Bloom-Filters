"""
GPU implementation that performs the exact same math as iterative_vectors.py.
Vocabulary = list(tf_idfs.keys()). Edges built by scanning corpus like CPU generate_vector.
Numerator = scatter-add of tfidf * V_prev; denominator = neighbor slot count; V_raw = numerator/denom;
then normalize_vector_dimensions (row norm + robust scaling) as CPU. No blending, no 0.json.
"""
from tqdm import tqdm
import sys
import time
import os
import json
import numpy as np

# CUDA path for Windows
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
if os.name == "nt" and os.path.isdir(_cuda_bin):
    os.environ["PATH"] = _cuda_bin + os.pathsep + os.environ.get("PATH", "")
    if "CUDA_PATH" not in os.environ:
        os.environ["CUDA_PATH"] = os.path.dirname(_cuda_bin)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_cuda_bin)

import cupy as cp

with open('data/fairytales_word_tf-idfs.json', 'r') as f:
    tf_idfs = json.load(f)
with open('data/fairytales_word_bloom-filters.json', 'r') as f:
    bloom_filters = json.load(f)
with open('data/fairytales_tokenized.json', 'r') as f:
    tokenized_corpus = json.load(f)

# Vocabulary exactly as CPU: list(tf_idfs.keys())
vocab = list(tf_idfs.keys())
word_to_idx = {w: i for i, w in enumerate(vocab)}
n_words = len(vocab)

# bits from first word that has a bloom filter
bits = 200
for w in vocab:
    if w in bloom_filters:
        bits = len(bloom_filters[w])
        break

# Rescale bloom to [-1, 1] like CPU
def rescale_bloom_filter():
    for word in list(bloom_filters.keys()):
        bloom_filters[word] = np.array(bloom_filters[word], dtype=np.float64) * 2 - 1

rescale_bloom_filter()
bloom_set = set(bloom_filters.keys())

def build_edges_iter0_and_all(deltas, vocab_set, bloom_set, tf_idfs, tokenized_corpus):
    edge_dst_all = []
    edge_src_all = []
    edge_w_all = []
    edge_dst_iter0 = []
    edge_src_iter0 = []
    edge_w_iter0 = []
    for sentence in tqdm(tokenized_corpus, desc="Building edges", leave=False):
        for p in range(len(sentence)):
            center = sentence[p]
            if center not in vocab_set:
                continue
            dst_idx = word_to_idx[center]
            for delta in deltas:
                q = p + delta
                if q < 0 or q >= len(sentence):
                    continue
                neighbor = sentence[q]
                if neighbor not in vocab_set:
                    continue
                w = tf_idfs.get(center, {}).get(neighbor, 0.0)
                src_idx = word_to_idx[neighbor]
                edge_dst_all.append(dst_idx)
                edge_src_all.append(src_idx)
                edge_w_all.append(w)
                if neighbor in bloom_set:
                    edge_dst_iter0.append(dst_idx)
                    edge_src_iter0.append(src_idx)
                    edge_w_iter0.append(w)
    return (
        (np.array(edge_dst_iter0), np.array(edge_src_iter0), np.array(edge_w_iter0, dtype=np.float64)),
        (np.array(edge_dst_all), np.array(edge_src_all), np.array(edge_w_all, dtype=np.float64)),
    )

def scatter_add_numerator(V_prev, edge_src, edge_dst, edge_w, n_words, bits):
    """Numerator = sum over edges of edge_w * V_prev[edge_src]. Uses scatter_add (GPU)."""
    V_diff = cp.zeros((n_words, bits), dtype=cp.float64)
    n_edges = edge_src.size
    BATCH = 2**20
    for start in range(0, n_edges, BATCH):
        end = min(start + BATCH, n_edges)
        batch_src = edge_src[start:end]
        batch_dst = edge_dst[start:end]
        batch_w = edge_w[start:end]
        gathered = V_prev[batch_src]
        weighted = gathered * batch_w[:, cp.newaxis]
        for d in range(bits):
            cp.add.at(V_diff[:, d], batch_dst, weighted[:, d])
    return V_diff

def normalize_vector_dimensions_gpu(vectors_cp):
    """Same as CPU: row norm then robust scaling (median, IQR). vectors_cp: (n_words, bits)."""
    norms = cp.linalg.norm(vectors_cp, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors_cp = vectors_cp / norms
    med = cp.median(vectors_cp, axis=0)
    q75 = cp.percentile(vectors_cp, 75, axis=0)
    q25 = cp.percentile(vectors_cp, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1
    vectors_cp = (vectors_cp - med) / iqr
    return vectors_cp

if __name__ == '__main__':
    NEIGHBORHOOD_SIZE = int(os.environ.get('NEIGHBORHOOD_SIZE', '4'))
    NUM_ITERATIONS = int(os.environ.get('ITERATIONS', '400'))
    deltas = [i for i in range(-NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE + 1) if i != 0]
    checkpoint_iterations = [0, 1, 4, 9, 24, 49, 74, 99, 149]
    checkpoint_iterations = [i for i in checkpoint_iterations if i < NUM_ITERATIONS]
    if NUM_ITERATIONS - 1 not in checkpoint_iterations:
        checkpoint_iterations.append(NUM_ITERATIONS - 1)

    os.makedirs('data/iterative_vectors', exist_ok=True)
    vocab_set = set(vocab)

    (edge_dst_i0, edge_src_i0, edge_w_i0), (edge_dst_all, edge_src_all, edge_w_all) = build_edges_iter0_and_all(
        deltas, vocab_set, bloom_set, tf_idfs, tokenized_corpus
    )

    denom_iter0 = np.bincount(edge_dst_i0, minlength=n_words).astype(np.float64)
    denom_iter0[denom_iter0 == 0] = 1
    denom_all = np.bincount(edge_dst_all, minlength=n_words).astype(np.float64)
    denom_all[denom_all == 0] = 1

    denom_iter0_cp = cp.asarray(denom_iter0)
    denom_all_cp = cp.asarray(denom_all)

    # Initial V for iter 0: bloom filter vectors for vocab (neighbors in scatter); words not in bloom get zeros
    V_prev = cp.zeros((n_words, bits), dtype=cp.float64)
    for i, w in enumerate(vocab):
        if w in bloom_filters:
            V_prev[i] = cp.asarray(bloom_filters[w], dtype=cp.float64)

    edge_src_i0_cp = cp.asarray(edge_src_i0)
    edge_dst_i0_cp = cp.asarray(edge_dst_i0)
    edge_w_i0_cp = cp.asarray(edge_w_i0)
    edge_src_all_cp = cp.asarray(edge_src_all)
    edge_dst_all_cp = cp.asarray(edge_dst_all)
    edge_w_all_cp = cp.asarray(edge_w_all)

    start_total = time.time()

    for iteration in range(NUM_ITERATIONS):
        iter_start = time.time()
        if iteration == 0:
            edge_dst_cp = edge_dst_i0_cp
            edge_src_cp = edge_src_i0_cp
            edge_w_cp = edge_w_i0_cp
            denom_cp = denom_iter0_cp
        else:
            edge_dst_cp = edge_dst_all_cp
            edge_src_cp = edge_src_all_cp
            edge_w_cp = edge_w_all_cp
            denom_cp = denom_all_cp

        numerator = scatter_add_numerator(V_prev, edge_src_cp, edge_dst_cp, edge_w_cp, n_words, bits)
        denom_expanded = cp.expand_dims(denom_cp, 1)
        V_raw = numerator / denom_expanded

        V_prev = normalize_vector_dimensions_gpu(V_raw)

        elapsed = time.time() - iter_start
        print(f"Iteration {iteration} took {elapsed:.2f} s")
        if iteration in checkpoint_iterations:
            V_save = cp.asnumpy(V_prev).astype(np.float64)
            out = {vocab[j]: V_save[j].tolist() for j in range(n_words)}
            path = os.path.join('data/iterative_vectors', f'window_{NEIGHBORHOOD_SIZE}_iter_{iteration}_v3_{bits}bit.json')
            with open(path, 'w') as f:
                json.dump(out, f, indent=4)
            print(f"Saved {path}")

    print(f"Total time: {time.time() - start_total:.2f} s")
