"""
TF-IDF weighted neighbor diffusion over vocabulary from tf_idfs.
Initialization: Bloom filters rescaled to [-1, 1]; words without Bloom get zeros.
Numerator = scatter-add of (tfidf * V_prev) over edges; denominator = neighbor slot count.
Update: V_raw = numerator / denom; then row L2 norm + robust scaling (median / IQR).
Iteration 0 uses only Bloom-existing neighbors; later iterations use all vocab neighbors.
Algorithm matches iterative_vectors.py (CPU reference).
"""
import time
import os
import json
import numpy as np

if os.environ.get("ADD_CUDA_BIN", "0") == "1":
    _cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
    if os.name == "nt" and os.path.isdir(_cuda_bin):
        os.environ["PATH"] = _cuda_bin + os.pathsep + os.environ.get("PATH", "")
        if "CUDA_PATH" not in os.environ:
            os.environ["CUDA_PATH"] = os.path.dirname(_cuda_bin)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_cuda_bin)

import cupy as cp
import cupyx

# If cp.scatter_add supports 2D indexed updates, we use it; otherwise fallback to cp.add.at per dimension.
SCATTER_ADD = getattr(cp, "scatter_add", None)
if SCATTER_ADD is None:
    SCATTER_ADD = getattr(cupyx, "scatter_add", None)

# -----------------------------------------------------------------------------
# Paths (override via env or keep defaults for reproducibility)
# -----------------------------------------------------------------------------
TFIDF_PATH = os.environ.get("TFIDF_PATH", "data/fairytales_word_tf-idfs.json")
BLOOM_PATH = os.environ.get("BLOOM_PATH", "data/fairytales_word_bloom-filters.json")
TOKENIZED_PATH = os.environ.get("TOKENIZED_PATH", "data/fairytales_tokenized.json")
OUT_DIR = os.environ.get("OUT_DIR", "data/iterative_vectors")

with open(TFIDF_PATH, "r") as f:
    tf_idfs = json.load(f)
with open(BLOOM_PATH, "r") as f:
    bloom_filters = json.load(f)
with open(TOKENIZED_PATH, "r") as f:
    tokenized_corpus = json.load(f)

vocab = list(tf_idfs.keys())
word_to_idx = {w: i for i, w in enumerate(vocab)}
n_words = len(vocab)

bits = 200
for w in vocab:
    if w in bloom_filters:
        bits = len(bloom_filters[w])
        break

# Bloom init in [-1, 1]
def rescale_bloom_filter():
    for word in list(bloom_filters.keys()):
        bloom_filters[word] = np.array(bloom_filters[word], dtype=np.float64) * 2 - 1

rescale_bloom_filter()
bloom_set = set(bloom_filters.keys())


def env_flag(name, default="1"):
    raw = os.environ.get(name, default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def env_positive_float(name, default):
    raw = str(os.environ.get(name, str(default))).strip()
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return value if value > 0 else float(default)


def build_edges_iter0_and_all(
    deltas,
    vocab_set,
    bloom_set,
    tf_idfs,
    tokenized_corpus,
    progress_every_seconds=5.0,
    enable_progress=True,
):
    """One-pass corpus scan, O(#tokens * window). Returns (iter0_edges, all_edges)."""
    edge_dst_all = []
    edge_src_all = []
    edge_w_all = []
    edge_dst_iter0 = []
    edge_src_iter0 = []
    edge_w_iter0 = []
    total_sentences = len(tokenized_corpus)
    edge_build_started = time.time()
    last_progress_log = edge_build_started

    for sent_idx, sentence in enumerate(tokenized_corpus, start=1):
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
                wt = tf_idfs.get(center, {}).get(neighbor, 0.0)
                src_idx = word_to_idx[neighbor]
                edge_dst_all.append(dst_idx)
                edge_src_all.append(src_idx)
                edge_w_all.append(wt)
                if neighbor in bloom_set:
                    edge_dst_iter0.append(dst_idx)
                    edge_src_iter0.append(src_idx)
                    edge_w_iter0.append(wt)

        if enable_progress:
            now = time.time()
            should_log = (
                sent_idx == 1
                or sent_idx == total_sentences
                or (now - last_progress_log) >= progress_every_seconds
            )
            if should_log:
                elapsed = max(0.0, now - edge_build_started)
                if total_sentences > 0:
                    pct = 100.0 * sent_idx / total_sentences
                else:
                    pct = 100.0
                eta = 0.0
                if sent_idx > 0 and total_sentences > sent_idx and elapsed > 0:
                    eta = elapsed / sent_idx * (total_sentences - sent_idx)
                print(
                    "phase build_edges progress "
                    f"{sent_idx}/{total_sentences} ({pct:.1f}%) "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s "
                    f"edges_all={len(edge_src_all)} edges_iter0={len(edge_src_iter0)}",
                    flush=True,
                )
                last_progress_log = now

    return (
        (
            np.array(edge_dst_iter0, dtype=np.int32),
            np.array(edge_src_iter0, dtype=np.int32),
            np.array(edge_w_iter0, dtype=np.float64),
        ),
        (
            np.array(edge_dst_all, dtype=np.int32),
            np.array(edge_src_all, dtype=np.int32),
            np.array(edge_w_all, dtype=np.float64),
        ),
    )


def scatter_add_numerator(V_prev, edge_src, edge_dst, edge_w, n_words, bits, BATCH, progress_ctx=None):
    """Numerator = sum over edges of edge_w * V_prev[edge_src]. float64 accumulation."""
    V_diff = cp.zeros((n_words, bits), dtype=cp.float64)
    col_template = cp.arange(bits, dtype=cp.int32)
    n_edges = edge_src.size
    total_chunks = (n_edges + BATCH - 1) // BATCH if n_edges > 0 else 0

    ctx = progress_ctx or {}
    progress_enabled = bool(ctx.get("enabled", True))
    progress_interval = float(ctx.get("interval_seconds", 5.0) or 5.0)
    if progress_interval <= 0:
        progress_interval = 5.0
    iter_label = ctx.get("iter")
    stage_label = str(ctx.get("stage", "scatter_add"))
    stage_started = time.time()
    last_progress_log = stage_started

    for chunk_idx, start in enumerate(range(0, n_edges, BATCH), start=1):
        end = min(start + BATCH, n_edges)
        batch_src = edge_src[start:end]
        batch_dst = edge_dst[start:end]
        batch_w = edge_w[start:end]
        gathered = V_prev[batch_src]
        weighted = gathered * batch_w[:, cp.newaxis]
        used_scatter_add = False
        if SCATTER_ADD is not None:
            try:
                row_flat = cp.repeat(batch_dst, bits)
                col_flat = cp.tile(col_template, batch_dst.size)
                SCATTER_ADD(V_diff, (row_flat, col_flat), weighted.ravel())
                used_scatter_add = True
            except TypeError:
                used_scatter_add = False
        if not used_scatter_add:
            for d in range(bits):
                cp.add.at(V_diff[:, d], batch_dst, weighted[:, d])

        if progress_enabled and iter_label is not None and total_chunks > 0:
            now = time.time()
            should_log = (
                chunk_idx == 1
                or chunk_idx == total_chunks
                or (now - last_progress_log) >= progress_interval
            )
            if should_log:
                elapsed = max(0.0, now - stage_started)
                pct = 100.0 * chunk_idx / total_chunks
                eta = 0.0
                if chunk_idx > 0 and total_chunks > chunk_idx and elapsed > 0:
                    eta = elapsed / chunk_idx * (total_chunks - chunk_idx)
                print(
                    f"iter {iter_label} stage {stage_label} progress "
                    f"{chunk_idx}/{total_chunks} ({pct:.1f}%) "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s edges={n_edges}",
                    flush=True,
                )
                last_progress_log = now

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


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration (env overrides for sweeps; defaults for paper)
    # -------------------------------------------------------------------------
    NEIGHBORHOOD_SIZE = int(os.environ.get("NEIGHBORHOOD_SIZE", "4"))
    ITERATIONS = int(os.environ.get("ITERATIONS", "400"))
    SAVE_EVERY_ITER = os.environ.get("SAVE_EVERY_ITER", "0") == "1"
    # Paper checkpoints; final iteration is always appended below (used only when SAVE_EVERY_ITER=0)
    CHECKPOINTS = [0, 1, 4, 9, 24, 49, 74, 99, 149]
    BATCH = int(os.environ.get("BATCH", str(2**20)))

    deltas = [i for i in range(-NEIGHBORHOOD_SIZE, NEIGHBORHOOD_SIZE + 1) if i != 0]
    checkpoint_iterations = [i for i in CHECKPOINTS if i < ITERATIONS]
    if ITERATIONS - 1 not in checkpoint_iterations:
        checkpoint_iterations.append(ITERATIONS - 1)

    os.makedirs(OUT_DIR, exist_ok=True)
    if SAVE_EVERY_ITER:
        full_iter_dir = os.path.join(OUT_DIR, f"v3_full_window_{NEIGHBORHOOD_SIZE}")
        os.makedirs(full_iter_dir, exist_ok=True)
    vocab_set = set(vocab)

    progress_log_interval = env_positive_float("PROGRESS_LOG_INTERVAL", 5.0)
    enable_build_progress = env_flag("BUILD_EDGES_PROGRESS", "1")
    enable_scatter_progress = env_flag("SCATTER_PROGRESS", "1")

    print(
        f"phase init corpus={os.path.basename(TOKENIZED_PATH)} vocab={n_words} bits={bits} "
        f"N={NEIGHBORHOOD_SIZE} iters={ITERATIONS} batch={BATCH}",
        flush=True,
    )

    edge_build_start = time.time()
    print("phase build_edges start", flush=True)
    (edge_dst_i0, edge_src_i0, edge_w_i0), (edge_dst_all, edge_src_all, edge_w_all) = build_edges_iter0_and_all(
        deltas,
        vocab_set,
        bloom_set,
        tf_idfs,
        tokenized_corpus,
        progress_every_seconds=progress_log_interval,
        enable_progress=enable_build_progress,
    )
    print(
        "phase build_edges done "
        f"iter0_edges={edge_src_i0.size} all_edges={edge_src_all.size} "
        f"elapsed={time.time() - edge_build_start:.2f}s",
        flush=True,
    )

    # Avoid division by zero for words with no valid neighbors
    print("phase denominators start", flush=True)
    denom_iter0 = np.bincount(edge_dst_i0, minlength=n_words).astype(np.float64)
    denom_iter0[denom_iter0 == 0] = 1
    denom_all = np.bincount(edge_dst_all, minlength=n_words).astype(np.float64)
    denom_all[denom_all == 0] = 1
    print("phase denominators done", flush=True)

    print("phase transfer_to_gpu start", flush=True)
    denom_iter0_cp = cp.asarray(denom_iter0)
    denom_all_cp = cp.asarray(denom_all)

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
    print("phase transfer_to_gpu done", flush=True)

    start_total = time.time()
    print("phase iteration_loop start", flush=True)
    iter_stage_log = os.environ.get("ITER_STAGE_LOG", "1") == "1"

    for iteration in range(ITERATIONS):
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

        if iter_stage_log:
            print(f"iter {iteration} stage scatter_add start", flush=True)

        scatter_start = time.time()
        numerator = scatter_add_numerator(
            V_prev,
            edge_src_cp,
            edge_dst_cp,
            edge_w_cp,
            n_words,
            bits,
            BATCH,
            progress_ctx={
                "enabled": enable_scatter_progress,
                "interval_seconds": progress_log_interval,
                "iter": iteration,
                "stage": "scatter_add",
            },
        )
        scatter_elapsed = time.time() - scatter_start
        if iter_stage_log:
            print(f"iter {iteration} stage scatter_add {scatter_elapsed:.2f}s", flush=True)

        if iter_stage_log:
            print(f"iter {iteration} stage divide start", flush=True)
        divide_start = time.time()
        denom_expanded = cp.expand_dims(denom_cp, 1)
        V_raw = numerator / denom_expanded
        divide_elapsed = time.time() - divide_start
        if iter_stage_log:
            print(f"iter {iteration} stage divide {divide_elapsed:.2f}s", flush=True)

        # Optional: dump row-normalized (pre-robust) vectors for IQR debug (stability_debug.py Step 3)
        if os.environ.get("PREROBUST_DUMP", "0") == "1" and iteration in (0, 1, 2, 10, 50):
            norms = cp.linalg.norm(V_raw, axis=1, keepdims=True)
            norms[norms == 0] = 1
            V_row_norm = V_raw / norms
            path_pre = os.path.join(OUT_DIR, f"window_{NEIGHBORHOOD_SIZE}_iter_{iteration}_v3_{bits}bit_prerobust.json")
            V_pre_save = cp.asnumpy(V_row_norm).astype(np.float32)
            out_pre = {vocab[j]: V_pre_save[j].tolist() for j in range(n_words)}
            with open(path_pre, "w") as f:
                json.dump(out_pre, f)
            print(f"  saved {path_pre}", flush=True)

        if iter_stage_log:
            print(f"iter {iteration} stage normalize start", flush=True)
        normalize_start = time.time()
        V_new = normalize_vector_dimensions_gpu(V_raw)
        normalize_elapsed = time.time() - normalize_start
        if iter_stage_log:
            print(f"iter {iteration} stage normalize {normalize_elapsed:.2f}s", flush=True)

        if iter_stage_log:
            print(f"iter {iteration} stage blend start", flush=True)
        blend_start = time.time()
        alpha = float(os.environ.get("ALPHA", "1.0"))
        if alpha < 1.0:
            V_prev = alpha * V_new + (1.0 - alpha) * V_prev
        else:
            V_prev = V_new
        blend_elapsed = time.time() - blend_start
        if iter_stage_log:
            print(f"iter {iteration} stage blend {blend_elapsed:.2f}s", flush=True)

        elapsed = time.time() - iter_start
        print(f"iter {iteration} total {elapsed:.2f}s", flush=True)

        if SAVE_EVERY_ITER:
            # Save every iteration as numeric file for Δ=1 stability comparison (vocab order consistent, float32).
            V_save = cp.asnumpy(V_prev).astype(np.float32)
            out = {vocab[j]: V_save[j].tolist() for j in range(n_words)}
            path = os.path.join(full_iter_dir, f"{iteration}.json")
            with open(path, "w") as f:
                json.dump(out, f)
            if iteration % 50 == 0 or iteration == ITERATIONS - 1:
                print(f"  saved {path}", flush=True)
        elif iteration in checkpoint_iterations:
            # Serialize float32 to reduce file size; computation remains float64.
            V_save = cp.asnumpy(V_prev).astype(np.float32)
            out = {vocab[j]: V_save[j].tolist() for j in range(n_words)}
            path = os.path.join(OUT_DIR, f"window_{NEIGHBORHOOD_SIZE}_iter_{iteration}_v3_{bits}bit.json")
            with open(path, "w") as f:
                json.dump(out, f)
            print(f"  saved {path}", flush=True)

    print(f"Total time: {time.time() - start_total:.2f}s", flush=True)
