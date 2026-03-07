"""dump_probe_edges.py

Small helper to export per-occurrence edge triples (src,dst,w) from your GPU pipeline
into an .npz file consumable by iter1_numerator_forensics.py.

Drop this into your repo (e.g., debug/dump_probe_edges.py) and call after building edges.

It intentionally does *not* rebuild edges; it only serializes what your pipeline produced.

Example (NumPy/CuPy):

    from dump_probe_edges import dump_edges_npz

    dump_edges_npz(
        out_path='debug/edges_iter1.npz',
        edge_src=edge_src,  # numpy or cupy 1D
        edge_dst=edge_dst,  # numpy or cupy 1D
        edge_w=edge_w,      # numpy or cupy 1D
        vocab=idx_to_word,  # list[str] (must match indexing)
    )

Optionally export only probe destinations to keep files small:

    dump_edges_npz(
        out_path='debug/edges_iter1_probe.npz',
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_w=edge_w,
        vocab=idx_to_word,
        probe_dst_ids=[word_to_idx['king'], word_to_idx['man'], word_to_idx['long']],
    )
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np


def _to_numpy_1d(x: Any) -> np.ndarray:
    """Convert numpy/cupy/torch 1D tensor/array to numpy 1D."""
    # CuPy
    if hasattr(x, "get") and callable(getattr(x, "get")):
        x = x.get()
    # PyTorch
    if hasattr(x, "detach") and callable(getattr(x, "detach")):
        x = x.detach().cpu().numpy()
    # NumPy
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    return x


def dump_edges_npz(
    out_path: str,
    edge_src: Any,
    edge_dst: Any,
    edge_w: Any,
    vocab: Optional[List[str]] = None,
    probe_dst_ids: Optional[Iterable[int]] = None,
) -> None:
    src = _to_numpy_1d(edge_src).astype(np.int64, copy=False)
    dst = _to_numpy_1d(edge_dst).astype(np.int64, copy=False)
    w = _to_numpy_1d(edge_w)

    if not (src.shape == dst.shape == w.shape):
        raise ValueError(f"src/dst/w shape mismatch: {src.shape} vs {dst.shape} vs {w.shape}")

    if probe_dst_ids is not None:
        probe_dst_ids = list(probe_dst_ids)
        mask = np.isin(dst, np.asarray(probe_dst_ids, dtype=np.int64))
        src = src[mask]
        dst = dst[mask]
        w = w[mask]

    if vocab is None:
        np.savez_compressed(out_path, src=src, dst=dst, w=w)
    else:
        vocab_arr = np.asarray(vocab, dtype=object)
        np.savez_compressed(out_path, src=src, dst=dst, w=w, vocab=vocab_arr)

    print(f"Wrote {out_path} with {src.shape[0]} edges")
