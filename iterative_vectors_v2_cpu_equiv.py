"""
GPU implementation that reproduces the historical CPU run as closely as possible.

Reference semantics:
- fixed vocabulary = sorted(keys(data/iterative_vectors/0.json))
- start from V_prev loaded from 0.json
- denominator counts every in-bounds neighbor slot whose token exists in vocab
- numerator adds tfidf(center, neighbor) * V_prev[neighbor] only when tfidf != 0
- normalize once per iteration with the same row-L2 + robust-column-scaling schedule

This script does NOT regenerate preprocessing artifacts and does NOT introduce
any algorithm changes such as row-stochastic weighting, alpha mixing, restart,
momentum, or extra normalization passes.
"""

from __future__ import annotations

import json
import os
import time

_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
if os.name == "nt" and os.path.isdir(_cuda_bin):
    os.environ["PATH"] = _cuda_bin + os.pathsep + os.environ.get("PATH", "")
    if "CUDA_PATH" not in os.environ:
        os.environ["CUDA_PATH"] = os.path.dirname(_cuda_bin)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_cuda_bin)

import cupy as cp
import numpy as np
from tqdm import tqdm


DATA_DIR = "data"
ITERATIVE_DIR = os.path.join(DATA_DIR, "iterative_vectors")
TOKENIZED_PATH = os.path.join(DATA_DIR, "fairytales_tokenized.json")
TFIDF_PATH = os.path.join(DATA_DIR, "fairytales_word_tf-idfs.json")
INIT_PATH = os.path.join(ITERATIVE_DIR, "0.json")

# Historical run appears to use [-4, -3, -2, -1, 1, 2, 3, 4].
NEIGHBORHOOD_SIZE = int(os.environ.get("NEIGHBORHOOD_SIZE", "4"))
ITERATIONS = int(os.environ.get("ITERATIONS", "400"))
BATCH_EDGES = int(os.environ.get("BATCH_EDGES", "250000"))
OUTPUT_TEMPLATE = os.environ.get(
    "OUTPUT_TEMPLATE",
    "window_{window}_iter_{iteration}_cpu_equiv_32bit.json",
)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def normalize_vector_dimensions_gpu(vectors: cp.ndarray) -> cp.ndarray:
    """CPU-equivalent normalize_vector_dimensions using GPU arrays."""
    norms = cp.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    med = cp.median(vectors, axis=0)
    q75 = cp.percentile(vectors, 75, axis=0)
    q25 = cp.percentile(vectors, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1
    return (vectors - med) / iqr


def precompute_denom_and_edges(tokenized_corpus, tf_idfs, word_to_idx, deltas):
    """
    Build CPU-equivalent denominator counts and numerator edge list.

    Denominator:
    - for every center occurrence and every in-bounds delta
    - if neighbor token exists in vocab: denom[center] += 1

    Numerator edges:
    - same scan, but only keep edges where tfidf != 0
    - edge is neighbor -> center with weight tfidf
    """
    n_words = len(word_to_idx)
    denom = np.zeros(n_words, dtype=np.float32)
    edge_src = []
    edge_dst = []
    edge_w = []

    for sentence in tqdm(tokenized_corpus, desc="Precomputing denom/edges", dynamic_ncols=True):
        sent_len = len(sentence)
        for center_pos, center in enumerate(sentence):
            center_idx = word_to_idx.get(center)
            if center_idx is None:
                continue

            tfidf_map = tf_idfs.get(center, {})
            for delta in deltas:
                neighbor_pos = center_pos + delta
                if neighbor_pos < 0 or neighbor_pos >= sent_len:
                    continue

                neighbor = sentence[neighbor_pos]
                neighbor_idx = word_to_idx.get(neighbor)
                if neighbor_idx is None:
                    continue

                # CPU denom increments when neighbor has a vector available.
                # In this strict-reproduction mode, vector exists iff neighbor ∈ vocab.
                denom[center_idx] += 1

                tfidf = tfidf_map.get(neighbor, 0.0)
                if tfidf != 0:
                    edge_src.append(neighbor_idx)
                    edge_dst.append(center_idx)
                    edge_w.append(float(tfidf))

    denom[denom == 0] = 1
    return (
        cp.asarray(denom, dtype=cp.float32)[:, None],
        cp.asarray(np.asarray(edge_src, dtype=np.int32)),
        cp.asarray(np.asarray(edge_dst, dtype=np.int32)),
        cp.asarray(np.asarray(edge_w, dtype=np.float32)),
    )


def scatter_add_numerator(v_prev: cp.ndarray, edge_src: cp.ndarray, edge_dst: cp.ndarray, edge_w: cp.ndarray) -> cp.ndarray:
    """Compute V_num with batched GPU scatter-add."""
    n_words, bits = v_prev.shape
    v_num = cp.zeros((n_words, bits), dtype=cp.float32)

    n_edges = int(edge_src.shape[0])
    for start in range(0, n_edges, BATCH_EDGES):
        stop = min(start + BATCH_EDGES, n_edges)
        src_batch = edge_src[start:stop]
        dst_batch = edge_dst[start:stop]
        w_batch = edge_w[start:stop]

        weighted = v_prev[src_batch] * w_batch[:, None]
        try:
            cp.scatter_add(v_num, dst_batch[:, None], weighted)
        except AttributeError:
            for dim in range(bits):
                cp.add.at(v_num[:, dim], dst_batch, weighted[:, dim])

    return v_num


def save_vectors(path: str, vocab, vectors_gpu: cp.ndarray) -> None:
    vectors_cpu = cp.asnumpy(vectors_gpu)
    payload = {word: vectors_cpu[i].tolist() for i, word in enumerate(vocab)}
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)


def main():
    os.makedirs(ITERATIVE_DIR, exist_ok=True)

    tokenized_corpus = load_json(TOKENIZED_PATH)
    tf_idfs = load_json(TFIDF_PATH)
    init_vectors = load_json(INIT_PATH)

    vocab = sorted(init_vectors.keys())
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    v_prev = cp.asarray(
        np.asarray([init_vectors[word] for word in vocab], dtype=np.float32)
    )
    bits = int(v_prev.shape[1])
    deltas = [i for i in range(-NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE + 1) if i != 0]

    print("Using strict CPU-equivalent GPU update")
    print(f"Vocab size: {len(vocab)}")
    print(f"Bits: {bits}")
    print(f"Deltas: {deltas}")

    denom, edge_src, edge_dst, edge_w = precompute_denom_and_edges(
        tokenized_corpus=tokenized_corpus,
        tf_idfs=tf_idfs,
        word_to_idx=word_to_idx,
        deltas=deltas,
    )

    print(f"Denominator rows: {denom.shape[0]}")
    print(f"Numerator edges: {int(edge_src.shape[0])}")

    # Save iteration 0 in the new format for direct comparison.
    iter0_path = os.path.join(
        ITERATIVE_DIR,
        OUTPUT_TEMPLATE.format(window=NEIGHBORHOOD_SIZE, iteration=0),
    )
    save_vectors(iter0_path, vocab, v_prev)
    print(f"Saved iteration 0 to {iter0_path}")

    start_time = time.time()
    for iteration in range(1, ITERATIONS):
        iter_start = time.time()
        v_num = scatter_add_numerator(v_prev, edge_src, edge_dst, edge_w)
        v_raw = v_num / denom
        v_prev = normalize_vector_dimensions_gpu(v_raw).astype(cp.float32)

        out_path = os.path.join(
            ITERATIVE_DIR,
            OUTPUT_TEMPLATE.format(window=NEIGHBORHOOD_SIZE, iteration=iteration),
        )
        save_vectors(out_path, vocab, v_prev)

        elapsed = time.time() - start_time
        avg = elapsed / iteration
        remaining = avg * (ITERATIONS - iteration - 1)
        print(
            f"iter {iteration:03d}/{ITERATIONS - 1} "
            f"| iter_time={time.time() - iter_start:.2f}s "
            f"| elapsed={elapsed / 60:.2f}m "
            f"| eta={remaining / 60:.2f}m"
        )

    print(f"Finished in {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
