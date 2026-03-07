#!/usr/bin/env python3
"""iter1_numerator_forensics.py

Purpose
-------
Numerator-stage forensics for Iter-1 equivalence debugging:

- CPU-per-occurrence numerator log for probe words by scanning tokenized_corpus.
- Edge-list numerator log for the same probe words (from your GPU edge pipeline).
- Compare counts/weights/histograms to decide:
    (A) edge construction mismatch vs (B) accumulation dtype/order drift.

This script is intentionally *surgical* and does not change algorithm semantics.

Expected inputs
---------------
1) V_prev (iteration 0) vectors as JSON (ground-truth): data/iterative_vectors/0.json
   - The JSON key insertion-order is treated as the canonical vocab order.

2) tokenized_corpus: list[list[str]] (or equivalent) in .json/.pkl/.npy/.npz

3) tf_idfs: dict[str, dict[str, float]] in .json/.pkl

4) edge file (.npz) containing at least:
      - src: int32/int64 array shape (E,)
      - dst: int32/int64 array shape (E,)
      - w:   float32/float64 array shape (E,)
   Optional:
      - vocab: object array/list[str] giving idx->word for src/dst indices.
              If omitted, we assume vocab order == V_prev JSON key order.

Outputs
-------
Prints a compact report per probe word:
  - CPU slot stats (denom, nonzero slots, sum weights, top neighbors)
  - Edge stats (edge counts, sum weights, top sources)
  - Per-neighbor total weight map diffs (strongest signal for edge mismatch)
  - Numerator vector diffs under multiple accumulation variants

Usage
-----
python iter1_numerator_forensics.py \
  --vprev0 data/iterative_vectors/0.json \
  --tokenized data/tokenized_corpus.json \
  --tfidf data/tf_idfs.json \
  --edges debug/edges_iter1.npz \
  --probe king man long

Optional GPU tests (if cupy is installed and a CUDA GPU is available):
  --gpu

Notes
-----
- Denominator logic here matches the described historical CPU semantics:
  denom increments for every in-bounds neighbor *slot* where neighbor vector exists
  (even if TFIDF(w,n) is missing or 0.0).
- Numerator includes only TF-IDF-weighted neighbor vectors; missing TF-IDF -> 0.0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


# ------------------------------ helpers ------------------------------

def load_json_ordered(path: str) -> OrderedDict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def load_any(path: str) -> Any:
    """Best-effort loader.

    Supported:
      - .json: json.load
      - .pkl/.pickle: pickle.load
      - .npy: np.load(allow_pickle=True)
      - .npz: np.load (returns NpzFile)
      - .txt: lines (stripped)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    if ext == ".npy":
        return np.load(path, allow_pickle=True)
    if ext == ".npz":
        return np.load(path, allow_pickle=True)
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    raise ValueError(f"Unsupported extension for {path}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an == 0.0 or bn == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def topk_counter(counter: Counter, k: int = 20) -> List[Tuple[Any, int]]:
    return counter.most_common(k)


def topk_weight_map(weight_map: Dict[Any, float], k: int = 20) -> List[Tuple[Any, float]]:
    return sorted(weight_map.items(), key=lambda kv: kv[1], reverse=True)[:k]


def fmt_vec_stats(x: np.ndarray) -> str:
    return (
        f"min={float(np.min(x)):.6g} max={float(np.max(x)):.6g} "
        f"mean={float(np.mean(x)):.6g} l2={float(np.linalg.norm(x)):.6g}"
    )


# ------------------------------ data structures ------------------------------

@dataclass
class CpuSlotLog:
    denom_slots: int
    nonzero_slots: int
    sum_weights: float
    neighbor_freq_all: Counter
    neighbor_freq_nz: Counter
    neighbor_weight_sum: Dict[str, float]
    numerator_f32: np.ndarray  # (D,) float32
    numerator_f64: np.ndarray  # (D,) float64


@dataclass
class EdgeLog:
    edge_count_all: int
    edge_count_nz: int
    sum_edge_weights_nz: float
    src_freq_all: Counter
    src_freq_nz: Counter
    src_weight_sum: Dict[int, float]
    numerator_loop_f32: np.ndarray
    numerator_loop_f64: np.ndarray
    numerator_loop_f32_sorted: np.ndarray
    numerator_loop_f64_sorted: np.ndarray
    numerator_cupy_scatter_f32: np.ndarray | None
    numerator_cupy_scatter_f64: np.ndarray | None
    numerator_cupy_scatter_f32_sorted: np.ndarray | None
    numerator_cupy_scatter_f64_sorted: np.ndarray | None


# ------------------------------ core logic ------------------------------


def scan_corpus_cpu_slots(
    tokenized_corpus: Any,
    probes: List[str],
    deltas: List[int],
    vprev_words_set: set,
    word_to_idx: Dict[str, int],
    vprev_mat_f32: np.ndarray,
    tf_idfs: Dict[str, Dict[str, float]],
) -> Dict[str, CpuSlotLog]:
    """Scan tokenized_corpus once, collecting per-probe CPU-slot numerators."""

    # Normalize corpus shape: accept list of lists or numpy arrays of objects.
    if isinstance(tokenized_corpus, np.ndarray):
        # If it's an array of dtype=object containing lists.
        tokenized = tokenized_corpus.tolist()
    else:
        tokenized = tokenized_corpus

    if not isinstance(tokenized, list):
        raise TypeError("tokenized_corpus must be a list (of sentences) or an object array")

    dim = int(vprev_mat_f32.shape[1])
    vprev_mat_f64 = vprev_mat_f32.astype(np.float64, copy=False)

    probes_set = set(probes)
    acc: Dict[str, Dict[str, Any]] = {}
    for w in probes:
        acc[w] = {
            "denom": 0,
            "nonzero": 0,
            "sum_weights": 0.0,
            "freq_all": Counter(),
            "freq_nz": Counter(),
            "w_sum": defaultdict(float),
            "num32": np.zeros((dim,), dtype=np.float32),
            "num64": np.zeros((dim,), dtype=np.float64),
        }

    for sent_idx, sent in enumerate(tokenized):
        if sent is None:
            continue
        # Accept tokens as list[str] or np.ndarray[str]
        if isinstance(sent, np.ndarray):
            sent_list = sent.tolist()
        else:
            sent_list = sent
        if not isinstance(sent_list, list):
            raise TypeError(f"Sentence {sent_idx} is not a list; got {type(sent_list)}")

        L = len(sent_list)
        for p, tok in enumerate(sent_list):
            if tok not in probes_set:
                continue

            w = tok
            tfidf_w = tf_idfs.get(w, {})
            a = acc[w]

            for d in deltas:
                q = p + d
                if q < 0 or q >= L:
                    continue
                n = sent_list[q]
                if n not in vprev_words_set:
                    continue

                # Denominator counts every valid neighbor slot with a vector.
                a["denom"] += 1
                a["freq_all"][n] += 1

                weight = tfidf_w.get(n, 0.0)
                if weight == 0.0:
                    continue

                a["nonzero"] += 1
                a["sum_weights"] += float(weight)
                a["freq_nz"][n] += 1
                a["w_sum"][n] += float(weight)

                n_idx = word_to_idx.get(n)
                if n_idx is None:
                    # If V_prev set membership was true, this should not happen.
                    raise KeyError(f"Neighbor {n} not in word_to_idx")

                # Numerator accumulation:
                # float64 path
                a["num64"] += float(weight) * vprev_mat_f64[n_idx]

                # float32 path (force float32 scalar)
                w32 = np.float32(weight)
                a["num32"] += w32 * vprev_mat_f32[n_idx]

    out: Dict[str, CpuSlotLog] = {}
    for w in probes:
        a = acc[w]
        out[w] = CpuSlotLog(
            denom_slots=int(a["denom"]),
            nonzero_slots=int(a["nonzero"]),
            sum_weights=float(a["sum_weights"]),
            neighbor_freq_all=a["freq_all"],
            neighbor_freq_nz=a["freq_nz"],
            neighbor_weight_sum=dict(a["w_sum"]),
            numerator_f32=a["num32"],
            numerator_f64=a["num64"],
        )
    return out



def build_edge_logs(
    edges_npz_path: str,
    probes: List[str],
    idx_to_word: List[str],
    word_to_idx: Dict[str, int],
    vprev_mat_f32: np.ndarray,
    run_gpu: bool,
) -> Dict[str, EdgeLog]:
    """Compute per-probe edge stats and numerator variants."""

    z = np.load(edges_npz_path, allow_pickle=True)
    required = {"src", "dst", "w"}
    if not required.issubset(set(z.files)):
        raise KeyError(
            f"{edges_npz_path} must contain {sorted(required)}; found {sorted(z.files)}"
        )

    src = z["src"].astype(np.int64, copy=False)
    dst = z["dst"].astype(np.int64, copy=False)
    w = z["w"]

    # Optional vocab override
    if "vocab" in z.files:
        vocab = z["vocab"]
        # vocab might be numpy object array or list
        if isinstance(vocab, np.ndarray):
            vocab_list = [str(x) for x in vocab.tolist()]
        else:
            vocab_list = [str(x) for x in vocab]
        if vocab_list != idx_to_word:
            # Allow mismatch but warn; in that case, edge indices are relative to vocab_list.
            # We'll remap by building idx->word from edge vocab.
            print(
                "[WARN] edges vocab != V_prev vocab order. Using edges' vocab for src/dst decoding.",
                file=sys.stderr,
            )
            idx_to_word_edges = vocab_list
            word_to_idx_edges = {w_: i for i, w_ in enumerate(idx_to_word_edges)}
        else:
            idx_to_word_edges = idx_to_word
            word_to_idx_edges = word_to_idx
    else:
        idx_to_word_edges = idx_to_word
        word_to_idx_edges = word_to_idx

    dim = int(vprev_mat_f32.shape[1])
    vprev_mat_f64 = vprev_mat_f32.astype(np.float64, copy=False)

    # GPU (optional)
    cp = None
    cupyx = None
    vprev_gpu_f32 = None
    vprev_gpu_f64 = None
    if run_gpu:
        try:
            import cupy as cp  # type: ignore
            import cupyx  # type: ignore

            # Move vprev matrices once.
            vprev_gpu_f32 = cp.asarray(vprev_mat_f32)
            vprev_gpu_f64 = cp.asarray(vprev_mat_f64)
        except Exception as e:
            print(f"[WARN] --gpu requested but cupy import/GPU init failed: {e}", file=sys.stderr)
            run_gpu = False

    def cupy_scatter_sum_single_dst(
        src_i: np.ndarray,
        w_i: np.ndarray,
        dtype: Any,
        sorted_order: np.ndarray | None,
    ) -> np.ndarray:
        """Atomic scatter-add based accumulation into a single (1,dim) row.

        This tries to emulate the per-element atomic add pattern typical in scatter-add.
        """
        assert cp is not None and cupyx is not None

        if sorted_order is not None:
            src_i = src_i[sorted_order]
            w_i = w_i[sorted_order]

        src_cp = cp.asarray(src_i, dtype=cp.int64)
        w_cp = cp.asarray(w_i, dtype=cp.float64 if dtype == np.float64 else cp.float32)

        V_gpu = vprev_gpu_f64 if dtype == np.float64 else vprev_gpu_f32
        assert V_gpu is not None

        contrib = (w_cp[:, None] * V_gpu[src_cp]).astype(cp.float64 if dtype == np.float64 else cp.float32)

        out = cp.zeros((1, dim), dtype=cp.float64 if dtype == np.float64 else cp.float32)

        # Build indices with broadcasting.
        rows = cp.zeros((contrib.shape[0], 1), dtype=cp.int64)
        cols = cp.arange(dim, dtype=cp.int64)[None, :]

        try:
            cupyx.scatter_add(out, (rows, cols), contrib)
        except Exception:
            # Fallback: flatten and scatter into 1D.
            out1 = cp.zeros((dim,), dtype=out.dtype)
            idx = cp.tile(cp.arange(dim, dtype=cp.int64), reps=int(contrib.shape[0]))
            cupyx.scatter_add(out1, idx, contrib.reshape(-1))
            out = out1[None, :]

        return cp.asnumpy(out[0])

    out_logs: Dict[str, EdgeLog] = {}

    for w_probe in probes:
        if w_probe not in word_to_idx_edges:
            raise KeyError(f"Probe word '{w_probe}' not found in edge vocab")
        dst_idx = int(word_to_idx_edges[w_probe])

        mask = (dst == dst_idx)
        src_i = src[mask]
        w_i = w[mask]

        # Ensure w_i is float array
        if not isinstance(w_i, np.ndarray):
            w_i = np.asarray(w_i)

        # Nonzero mask
        nz_mask = (w_i != 0)
        src_nz = src_i[nz_mask]
        w_nz = w_i[nz_mask]

        src_freq_all = Counter(src_i.tolist())
        src_freq_nz = Counter(src_nz.tolist())

        src_weight_sum: Dict[int, float] = defaultdict(float)
        for s, ww in zip(src_nz.tolist(), w_nz.tolist()):
            src_weight_sum[int(s)] += float(ww)

        # Sequential-loop accumulations (deterministic w.r.t. input order)
        num_loop_f32 = np.zeros((dim,), dtype=np.float32)
        num_loop_f64 = np.zeros((dim,), dtype=np.float64)

        # Convert weights to Python float once for speed
        # (but keep separate float32 path)
        for s, ww in zip(src_nz, w_nz):
            s_int = int(s)
            # float64
            num_loop_f64 += float(ww) * vprev_mat_f64[s_int]
            # float32
            num_loop_f32 += np.float32(ww) * vprev_mat_f32[s_int]

        # Sorted-edge deterministic loop (sort by src id)
        order = np.argsort(src_nz, kind="mergesort")  # stable

        num_loop_f32_sorted = np.zeros((dim,), dtype=np.float32)
        num_loop_f64_sorted = np.zeros((dim,), dtype=np.float64)
        for idx in order:
            s_int = int(src_nz[idx])
            ww = w_nz[idx]
            num_loop_f64_sorted += float(ww) * vprev_mat_f64[s_int]
            num_loop_f32_sorted += np.float32(ww) * vprev_mat_f32[s_int]

        # Optional GPU atomic scatter-add
        num_cupy_f32 = None
        num_cupy_f64 = None
        num_cupy_f32_sorted = None
        num_cupy_f64_sorted = None
        if run_gpu:
            try:
                num_cupy_f32 = cupy_scatter_sum_single_dst(src_nz, w_nz, np.float32, sorted_order=None)
                num_cupy_f64 = cupy_scatter_sum_single_dst(src_nz, w_nz, np.float64, sorted_order=None)
                num_cupy_f32_sorted = cupy_scatter_sum_single_dst(src_nz, w_nz, np.float32, sorted_order=order)
                num_cupy_f64_sorted = cupy_scatter_sum_single_dst(src_nz, w_nz, np.float64, sorted_order=order)
            except Exception as e:
                print(f"[WARN] cupy scatter-add failed for '{w_probe}': {e}", file=sys.stderr)

        out_logs[w_probe] = EdgeLog(
            edge_count_all=int(src_i.shape[0]),
            edge_count_nz=int(src_nz.shape[0]),
            sum_edge_weights_nz=float(np.sum(w_nz.astype(np.float64))) if w_nz.size else 0.0,
            src_freq_all=src_freq_all,
            src_freq_nz=src_freq_nz,
            src_weight_sum=dict(src_weight_sum),
            numerator_loop_f32=num_loop_f32,
            numerator_loop_f64=num_loop_f64,
            numerator_loop_f32_sorted=num_loop_f32_sorted,
            numerator_loop_f64_sorted=num_loop_f64_sorted,
            numerator_cupy_scatter_f32=num_cupy_f32,
            numerator_cupy_scatter_f64=num_cupy_f64,
            numerator_cupy_scatter_f32_sorted=num_cupy_f32_sorted,
            numerator_cupy_scatter_f64_sorted=num_cupy_f64_sorted,
        )

    return out_logs



def compare_weight_maps(
    cpu_neighbor_weight_sum: Dict[str, float],
    edge_src_weight_sum: Dict[int, float],
    idx_to_word: List[str],
    topk: int = 20,
) -> Dict[str, Any]:
    """Compare CPU per-neighbor total weights to edge per-src total weights.

    This is the strongest signal for *edge construction equivalence*.
    """

    # Convert edge map from src_id -> word
    edge_word_map: Dict[str, float] = {}
    for sid, ws in edge_src_weight_sum.items():
        if sid < 0 or sid >= len(idx_to_word):
            edge_word_map[f"<OOB:{sid}>"] = ws
        else:
            edge_word_map[idx_to_word[sid]] = ws

    cpu_keys = set(cpu_neighbor_weight_sum.keys())
    edge_keys = set(edge_word_map.keys())

    missing_in_edges = sorted(cpu_keys - edge_keys)
    extra_in_edges = sorted(edge_keys - cpu_keys)

    # Compute per-word absolute diffs where present
    diffs: List[Tuple[str, float, float, float]] = []
    for k in sorted(cpu_keys & edge_keys):
        c = float(cpu_neighbor_weight_sum[k])
        e = float(edge_word_map[k])
        diffs.append((k, c, e, abs(c - e)))

    diffs_sorted = sorted(diffs, key=lambda t: t[3], reverse=True)

    # Also compute summary metrics
    total_cpu = float(sum(cpu_neighbor_weight_sum.values()))
    total_edge = float(sum(edge_word_map.values()))
    total_abs_diff = float(sum(abs(float(cpu_neighbor_weight_sum.get(k, 0.0)) - float(edge_word_map.get(k, 0.0))) for k in (cpu_keys | edge_keys)))

    return {
        "total_cpu_weight": total_cpu,
        "total_edge_weight": total_edge,
        "total_abs_weight_diff": total_abs_diff,
        "missing_in_edges_count": len(missing_in_edges),
        "extra_in_edges_count": len(extra_in_edges),
        "missing_in_edges_top": missing_in_edges[:topk],
        "extra_in_edges_top": extra_in_edges[:topk],
        "largest_diffs": diffs_sorted[:topk],  # (word, cpu, edge, absdiff)
    }



def report_probe(
    w: str,
    cpu: CpuSlotLog,
    edge: EdgeLog,
    idx_to_word: List[str],
) -> str:
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append(f"PROBE WORD: {w}")
    lines.append("-" * 88)

    # CPU summary
    lines.append("[CPU slot scan]")
    lines.append(f"  denom_slots          : {cpu.denom_slots}")
    lines.append(f"  nonzero_tfidf_slots   : {cpu.nonzero_slots}")
    lines.append(f"  sum_weights(nonzero)  : {cpu.sum_weights:.9g}")
    lines.append(f"  numerator_f64 stats   : {fmt_vec_stats(cpu.numerator_f64)}")
    lines.append(f"  numerator_f32 stats   : {fmt_vec_stats(cpu.numerator_f32)}")

    top_freq = topk_counter(cpu.neighbor_freq_all, 10)
    top_nz = topk_counter(cpu.neighbor_freq_nz, 10)
    top_wsum = topk_weight_map(cpu.neighbor_weight_sum, 10)

    lines.append("  top10 neighbors by freq (ALL slots):")
    lines.extend([f"    {nw!r}: {c}" for nw, c in top_freq])
    lines.append("  top10 neighbors by freq (NONZERO slots):")
    lines.extend([f"    {nw!r}: {c}" for nw, c in top_nz])
    lines.append("  top10 neighbors by total weight (NONZERO slots):")
    lines.extend([f"    {nw!r}: {ws:.9g}" for nw, ws in top_wsum])

    # Edge summary
    lines.append("\n[Edge list]")
    lines.append(f"  edge_count_all        : {edge.edge_count_all}")
    lines.append(f"  edge_count_nonzero_w  : {edge.edge_count_nz}")
    lines.append(f"  sum_edge_weights(nz)  : {edge.sum_edge_weights_nz:.9g}")

    top_src_freq = edge.src_freq_all.most_common(10)
    top_src_nz = edge.src_freq_nz.most_common(10)
    top_src_wsum = topk_weight_map({idx_to_word[s]: wsum for s, wsum in edge.src_weight_sum.items() if 0 <= s < len(idx_to_word)}, 10)

    lines.append("  top10 src by freq (ALL edges):")
    for sid, c in top_src_freq:
        wsrc = idx_to_word[sid] if 0 <= sid < len(idx_to_word) else f"<OOB:{sid}>"
        lines.append(f"    {wsrc!r} (id={sid}): {c}")
    lines.append("  top10 src by freq (NONZERO edges):")
    for sid, c in top_src_nz:
        wsrc = idx_to_word[sid] if 0 <= sid < len(idx_to_word) else f"<OOB:{sid}>"
        lines.append(f"    {wsrc!r} (id={sid}): {c}")
    lines.append("  top10 src by total weight (NONZERO edges):")
    lines.extend([f"    {wsrc!r}: {ws:.9g}" for wsrc, ws in top_src_wsum])

    # Strong edge-equivalence check: per-neighbor total weight map
    wm = compare_weight_maps(cpu.neighbor_weight_sum, edge.src_weight_sum, idx_to_word, topk=10)
    lines.append("\n[Edge vs CPU weight-map equivalence check]")
    lines.append(f"  total_cpu_weight      : {wm['total_cpu_weight']:.9g}")
    lines.append(f"  total_edge_weight     : {wm['total_edge_weight']:.9g}")
    lines.append(f"  total_abs_weight_diff : {wm['total_abs_weight_diff']:.9g}")
    lines.append(f"  missing_in_edges      : {wm['missing_in_edges_count']} (showing up to 10)")
    if wm["missing_in_edges_top"]:
        lines.append(f"    {wm['missing_in_edges_top']}")
    lines.append(f"  extra_in_edges        : {wm['extra_in_edges_count']} (showing up to 10)")
    if wm["extra_in_edges_top"]:
        lines.append(f"    {wm['extra_in_edges_top']}")
    lines.append("  largest per-neighbor |cpu_weight_sum - edge_weight_sum| (top 10):")
    for word, c, e, d in wm["largest_diffs"]:
        lines.append(f"    {word!r}: cpu={c:.9g} edge={e:.9g} absdiff={d:.9g}")

    # Numerator comparisons (these are only meaningful if weight-map matches)
    def vec_cmp(name: str, a: np.ndarray, b: np.ndarray) -> None:
        diff = a.astype(np.float64) - b.astype(np.float64)
        lines.append(
            f"  {name:<30} max_abs={float(np.max(np.abs(diff))):.9g} "
            f"cos={cosine(a.astype(np.float64), b.astype(np.float64)):.9g} "
            f"l2_diff={float(np.linalg.norm(diff)):.9g}"
        )

    lines.append("\n[Numerator vector comparisons]")
    vec_cmp("edge_loop_f64 vs cpu_f64", edge.numerator_loop_f64, cpu.numerator_f64)
    vec_cmp("edge_loop_f32 vs cpu_f32", edge.numerator_loop_f32, cpu.numerator_f32)
    vec_cmp("edge_loop_f64_sorted vs cpu_f64", edge.numerator_loop_f64_sorted, cpu.numerator_f64)
    vec_cmp("edge_loop_f32_sorted vs cpu_f32", edge.numerator_loop_f32_sorted, cpu.numerator_f32)

    if edge.numerator_cupy_scatter_f32 is not None:
        vec_cmp("cupy_scatter_f32 vs cpu_f32", edge.numerator_cupy_scatter_f32, cpu.numerator_f32)
    if edge.numerator_cupy_scatter_f64 is not None:
        vec_cmp("cupy_scatter_f64 vs cpu_f64", edge.numerator_cupy_scatter_f64, cpu.numerator_f64)
    if edge.numerator_cupy_scatter_f32_sorted is not None:
        vec_cmp("cupy_scatter_f32_sorted vs cpu_f32", edge.numerator_cupy_scatter_f32_sorted, cpu.numerator_f32)
    if edge.numerator_cupy_scatter_f64_sorted is not None:
        vec_cmp("cupy_scatter_f64_sorted vs cpu_f64", edge.numerator_cupy_scatter_f64_sorted, cpu.numerator_f64)

    return "\n".join(lines)


# ------------------------------ CLI ------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vprev0", required=True, help="Path to data/iterative_vectors/0.json")
    ap.add_argument("--tokenized", required=True, help="Path to tokenized_corpus (json/pkl/npy/npz)")
    ap.add_argument("--tfidf", required=True, help="Path to tf_idfs (json/pkl)")
    ap.add_argument("--edges", required=True, help="Path to edges_iter1.npz containing src,dst,w (and optional vocab)")
    ap.add_argument("--probe", nargs="+", default=["king", "man", "long"], help="Probe words")
    ap.add_argument(
        "--window",
        type=int,
        default=4,
        help="Window radius; deltas = [-window..-1, 1..window]",
    )
    ap.add_argument("--gpu", action="store_true", help="Also run cupy scatter-add variants (if available)")
    return ap.parse_args()



def main() -> int:
    args = parse_args()

    vprev0 = load_json_ordered(args.vprev0)
    idx_to_word = list(vprev0.keys())
    word_to_idx = {w: i for i, w in enumerate(idx_to_word)}

    # V_prev matrix
    first_vec = next(iter(vprev0.values()))
    dim = len(first_vec)
    vprev_mat_f32 = np.vstack([np.asarray(v, dtype=np.float32) for v in vprev0.values()])

    # Load tokenized + tfidf
    tokenized = load_any(args.tokenized)
    tf_idfs = load_any(args.tfidf)

    if not isinstance(tf_idfs, dict):
        raise TypeError(f"tf_idfs must be a dict; got {type(tf_idfs)}")

    # Deltas
    w = int(args.window)
    deltas = list(range(-w, 0)) + list(range(1, w + 1))

    probes = args.probe

    # CPU slot logs
    cpu_logs = scan_corpus_cpu_slots(
        tokenized_corpus=tokenized,
        probes=probes,
        deltas=deltas,
        vprev_words_set=set(vprev0.keys()),
        word_to_idx=word_to_idx,
        vprev_mat_f32=vprev_mat_f32,
        tf_idfs=tf_idfs,
    )

    # Edge logs
    edge_logs = build_edge_logs(
        edges_npz_path=args.edges,
        probes=probes,
        idx_to_word=idx_to_word,
        word_to_idx=word_to_idx,
        vprev_mat_f32=vprev_mat_f32,
        run_gpu=bool(args.gpu),
    )

    # Reports
    for w_probe in probes:
        print(report_probe(w_probe, cpu_logs[w_probe], edge_logs[w_probe], idx_to_word))

    print("\n" + "=" * 88)
    print("INTERPRETATION GUIDE")
    print("- If the weight-map equivalence check shows missing/extra neighbors or large diffs,")
    print("  you have an EDGE CONSTRUCTION / WEIGHT LOOKUP mismatch (case A).")
    print("- If weight-map diffs are ~0 (or within tiny float noise) but scatter-add differs")
    print("  while deterministic loop matches, it's ACCUMULATION ORDER/DTYPE drift (case B).")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
