from __future__ import annotations

import json
import os
from collections import OrderedDict

import numpy as np

_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
if os.name == "nt" and os.path.isdir(_cuda_bin):
    os.environ["PATH"] = _cuda_bin + os.pathsep + os.environ.get("PATH", "")
    if "CUDA_PATH" not in os.environ:
        os.environ["CUDA_PATH"] = os.path.dirname(_cuda_bin)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_cuda_bin)

import cupy as cp


DATA_DIR = "data"
ITER_DIR = os.path.join(DATA_DIR, "iterative_vectors")
INIT_PATH = os.path.join(ITER_DIR, "0.json")
GT1_PATH = os.path.join(ITER_DIR, "1.json")
TOKENIZED_PATH = os.path.join(DATA_DIR, "fairytales_tokenized.json")
TFIDF_PATH = os.path.join(DATA_DIR, "fairytales_word_tf-idfs.json")
NEIGHBORHOOD_SIZE = int(os.environ.get("NEIGHBORHOOD_SIZE", "4"))


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def normalize_rows_numpy(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def robust_scale_numpy(vectors: np.ndarray) -> np.ndarray:
    med = np.median(vectors, axis=0)
    q75 = np.percentile(vectors, 75, axis=0)
    q25 = np.percentile(vectors, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1
    return (vectors - med) / iqr


def build_sample_words(vocab_keys):
    sample = []
    sample.extend(vocab_keys[:10])
    sample.extend(vocab_keys[-10:])
    for word in ["king", "queen", "man", "woman"]:
        if word in vocab_keys and word not in sample:
            sample.append(word)
    return sample


def direct_cpu_sample(tokenized_corpus, tf_idfs, v_prev_dict, sample_words, deltas, gt1_dict):
    vocab_set = set(v_prev_dict.keys())
    bits = len(next(iter(v_prev_dict.values())))
    numerators = {}
    denoms = {}
    raw = {}
    row = {}

    for word in sample_words:
        numerator = np.zeros(bits, dtype=np.float64)
        denom = 0
        tfidf_map = tf_idfs.get(word, {})
        for sentence in tokenized_corpus:
            for center_pos, center in enumerate(sentence):
                if center != word:
                    continue
                for delta in deltas:
                    neighbor_pos = center_pos + delta
                    if neighbor_pos < 0 or neighbor_pos >= len(sentence):
                        continue
                    neighbor = sentence[neighbor_pos]
                    if neighbor in vocab_set:
                        denom += 1
                        tfidf = tfidf_map.get(neighbor, 0.0)
                        if tfidf != 0:
                            numerator += tfidf * np.asarray(v_prev_dict[neighbor], dtype=np.float64)
        denom_safe = 1 if denom == 0 else denom
        raw_vec = numerator / denom_safe
        row_norm = np.linalg.norm(raw_vec)
        row_vec = raw_vec if row_norm == 0 else raw_vec / row_norm

        denoms[word] = denom
        numerators[word] = numerator
        raw[word] = raw_vec
        row[word] = row_vec

    final = {word: np.asarray(gt1_dict[word], dtype=np.float64) for word in sample_words}
    return {
        "denom": denoms,
        "numerator": numerators,
        "raw": raw,
        "row": row,
        "final": final,
    }


def gpu_iteration1_full(tokenized_corpus, tf_idfs, vocab_keys, v_prev_np, deltas):
    word_to_idx = {word: i for i, word in enumerate(vocab_keys)}
    vocab_set = set(vocab_keys)
    n_words = len(vocab_keys)

    denom = np.zeros(n_words, dtype=np.float64)
    edge_src = []
    edge_dst = []
    edge_w = []

    for sentence in tokenized_corpus:
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
                denom[center_idx] += 1
                tfidf = tfidf_map.get(neighbor, 0.0)
                if tfidf != 0:
                    edge_src.append(neighbor_idx)
                    edge_dst.append(center_idx)
                    edge_w.append(float(tfidf))

    denom_safe = denom.copy()
    denom_safe[denom_safe == 0] = 1

    v_prev_gpu = cp.asarray(v_prev_np.astype(np.float32))
    edge_src_gpu = cp.asarray(np.asarray(edge_src, dtype=np.int32))
    edge_dst_gpu = cp.asarray(np.asarray(edge_dst, dtype=np.int32))
    edge_w_gpu = cp.asarray(np.asarray(edge_w, dtype=np.float32))

    v_num_gpu = cp.zeros_like(v_prev_gpu)
    weighted = v_prev_gpu[edge_src_gpu] * edge_w_gpu[:, None]
    try:
        cp.scatter_add(v_num_gpu, edge_dst_gpu[:, None], weighted)
    except AttributeError:
        for dim in range(v_prev_gpu.shape[1]):
            cp.add.at(v_num_gpu[:, dim], edge_dst_gpu, weighted[:, dim])

    v_num = cp.asnumpy(v_num_gpu).astype(np.float64)
    v_raw = v_num / denom_safe[:, None]
    v_row = normalize_rows_numpy(v_raw)
    v_final = robust_scale_numpy(v_row)

    return {
        "denom": denom,
        "numerator": v_num,
        "raw": v_raw,
        "row": v_row,
        "final": v_final,
    }


def stage_report(word, cpu_stage, gpu_stage):
    cpu_vec = np.asarray(cpu_stage, dtype=np.float64)
    gpu_vec = np.asarray(gpu_stage, dtype=np.float64)
    return f"max_abs_diff={max_abs_diff(cpu_vec, gpu_vec):.6e}, cosine={cosine_similarity(cpu_vec, gpu_vec):.6f}"


def main():
    v_prev_dict = load_json(INIT_PATH)
    gt1_dict = load_json(GT1_PATH)
    tokenized_corpus = load_json(TOKENIZED_PATH)
    tf_idfs = load_json(TFIDF_PATH)

    vocab_keys = list(v_prev_dict.keys())  # exact 0.json key order
    sample_words = build_sample_words(vocab_keys)
    deltas = [i for i in range(-NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE + 1) if i != 0]

    print("Sample words:")
    print(sample_words)
    print(f"Using deltas: {deltas}")

    cpu_sample = direct_cpu_sample(
        tokenized_corpus=tokenized_corpus,
        tf_idfs=tf_idfs,
        v_prev_dict=v_prev_dict,
        sample_words=sample_words,
        deltas=deltas,
        gt1_dict=gt1_dict,
    )

    v_prev_np = np.asarray([v_prev_dict[word] for word in vocab_keys], dtype=np.float64)
    gpu_full = gpu_iteration1_full(
        tokenized_corpus=tokenized_corpus,
        tf_idfs=tf_idfs,
        vocab_keys=vocab_keys,
        v_prev_np=v_prev_np,
        deltas=deltas,
    )
    word_to_idx = {word: i for i, word in enumerate(vocab_keys)}

    print("\nDetailed sample comparison:")
    first_mismatch_counts = {
        "denom": 0,
        "numerator": 0,
        "raw": 0,
        "row": 0,
        "final": 0,
        "none": 0,
    }

    tol = 1e-6
    for word in sample_words:
        idx = word_to_idx[word]
        denom_cpu = cpu_sample["denom"][word]
        denom_gpu = int(gpu_full["denom"][idx])
        num_cpu = cpu_sample["numerator"][word]
        num_gpu = gpu_full["numerator"][idx]
        raw_cpu = cpu_sample["raw"][word]
        raw_gpu = gpu_full["raw"][idx]
        row_cpu = cpu_sample["row"][word]
        row_gpu = gpu_full["row"][idx]
        final_cpu = cpu_sample["final"][word]
        final_gpu = gpu_full["final"][idx]

        if denom_cpu != denom_gpu:
            first = "denom"
        elif max_abs_diff(num_cpu, num_gpu) > tol:
            first = "numerator"
        elif max_abs_diff(raw_cpu, raw_gpu) > tol:
            first = "raw"
        elif max_abs_diff(row_cpu, row_gpu) > tol:
            first = "row"
        elif max_abs_diff(final_cpu, final_gpu) > tol:
            first = "final"
        else:
            first = "none"
        first_mismatch_counts[first] += 1

        print(f"\nWord: {word}")
        print(f"  first_mismatch_stage: {first}")
        print(f"  denom_cpu={denom_cpu}, denom_gpu={denom_gpu}")
        print(f"  numerator_cpu[:10]={num_cpu[:10].tolist()}")
        print(f"  numerator_gpu[:10]={num_gpu[:10].tolist()}")
        print(f"  numerator_cmp: {stage_report(word, num_cpu, num_gpu)}")
        print(f"  V_raw_cpu[:10]={raw_cpu[:10].tolist()}")
        print(f"  V_raw_gpu[:10]={raw_gpu[:10].tolist()}")
        print(f"  V_raw_cmp: {stage_report(word, raw_cpu, raw_gpu)}")
        print(f"  V_row_cpu[:10]={row_cpu[:10].tolist()}")
        print(f"  V_row_gpu[:10]={row_gpu[:10].tolist()}")
        print(f"  V_row_cmp: {stage_report(word, row_cpu, row_gpu)}")
        print(f"  V_final_cpu[:10]={final_cpu[:10].tolist()}")
        print(f"  V_final_gpu[:10]={final_gpu[:10].tolist()}")
        print(f"  V_final_cmp: {stage_report(word, final_cpu, final_gpu)}")

    print("\nFirst mismatch stage counts:")
    for stage, count in first_mismatch_counts.items():
        print(f"  {stage}: {count}")


if __name__ == "__main__":
    main()
