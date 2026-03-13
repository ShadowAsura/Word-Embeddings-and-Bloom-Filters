#!/usr/bin/env python3
"""
Train a Word2Vec (gensim) baseline on the existing tokenized corpus and export vectors
to the same JSON format as diffusion checkpoints.

Usage (PowerShell):
  python evaluation/train_word2vec_baseline.py --window 4 --epochs 50 --negative 5 --dim 32
  python evaluation/train_word2vec_baseline.py --window 6 --epochs 50 --dim auto --run-eval

Notes:
- For reproducibility, default workers=1. Gensim training is not fully deterministic with workers>1.
- Vocab is restricted to tf_idfs.keys() so it matches the diffusion vocab.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Gensim import (fail with a clear message if not installed)
try:
    from gensim.models import Word2Vec
except Exception as e:
    raise SystemExit(
        "gensim is required. Install with:\n"
        "  pip install gensim\n\n"
        f"Import error: {e}"
    )


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_dim_from_bloom(bloom_path: Path) -> Optional[int]:
    """Infer embedding dimension from bloom filter JSON: first list length."""
    try:
        bloom = load_json(bloom_path)
    except FileNotFoundError:
        return None
    for _, v in bloom.items():
        if isinstance(v, list) and len(v) > 0:
            return int(len(v))
    return None


def infer_dim_from_embedding_json(emb_path: Path) -> Optional[int]:
    """Infer dimension from a saved embedding JSON (word -> list[float])."""
    try:
        emb = load_json(emb_path)
    except FileNotFoundError:
        return None
    for _, vec in emb.items():
        if isinstance(vec, list) and len(vec) > 0:
            return int(len(vec))
    return None


def filter_sentences_to_vocab(
    sentences: List[List[str]], vocab_set: set[str]
) -> List[List[str]]:
    filtered: List[List[str]] = []
    for s in sentences:
        s2 = [t for t in s if t in vocab_set]
        if len(s2) >= 2:
            filtered.append(s2)
    return filtered


def export_vectors_json(
    model: Word2Vec, vocab_order: List[str], out_path: Path
) -> Tuple[int, int]:
    """
    Export vectors for vocab_order to JSON.
    Returns (kept_count, missing_count).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vectors: Dict[str, List[float]] = {}
    missing = 0
    for w in vocab_order:
        if w in model.wv:
            vec = model.wv[w].astype(np.float32, copy=False)
            vectors[w] = vec.tolist()
        else:
            missing += 1

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(vectors, f)

    return (len(vectors), missing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenized", default="data/fairytales_tokenized.json")
    ap.add_argument("--tfidf", default="data/fairytales_word_tf-idfs.json")
    ap.add_argument("--bloom", default="data/fairytales_word_bloom-filters.json")

    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--negative", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--sg", type=int, default=1, help="1=skip-gram, 0=CBOW")
    ap.add_argument("--sample", type=float, default=0.0, help="0 disables subsampling")
    ap.add_argument("--workers", type=int, default=1, help="Use 1 for determinism")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--dim",
        default="auto",
        help="Embedding dimension: int or 'auto'. "
             "Auto tries bloom length, else tries diffusion checkpoint iter_1, else 32.",
    )
    ap.add_argument(
        "--match-diffusion",
        default="",
        help="Optional: path to a diffusion embedding JSON to match dimension from (len of vectors).",
    )

    ap.add_argument(
        "--out",
        default="",
        help="Output JSON path. If empty, auto-names under data/word2vec/.",
    )

    ap.add_argument(
        "--run-eval",
        action="store_true",
        help="If set, run evaluation/evaluate_analogies.py on the exported JSON.",
    )
    ap.add_argument(
        "--questions",
        default="evaluation/analogy_questions.json",
        help="Analogy questions JSON (same format as your evaluator).",
    )

    args = ap.parse_args()

    tokenized_path = Path(args.tokenized)
    tfidf_path = Path(args.tfidf)
    bloom_path = Path(args.bloom)

    sentences: List[List[str]] = load_json(tokenized_path)
    tf_idfs: Dict[str, Dict[str, float]] = load_json(tfidf_path)

    vocab_order = list(tf_idfs.keys())  # match your diffusion baseline
    vocab_set = set(vocab_order)

    # Determine dimension
    dim: Optional[int] = None
    if args.dim != "auto":
        dim = int(args.dim)
    else:
        # 1) try bloom length
        dim = infer_dim_from_bloom(bloom_path)
        # 2) try user-specified diffusion path
        if dim is None and args.match_diffusion:
            dim = infer_dim_from_embedding_json(Path(args.match_diffusion))
        # 3) try common diffusion checkpoint path
        if dim is None:
            dim = infer_dim_from_embedding_json(
                Path("data/iterative_vectors/window_4_iter_1_v3_32bit.json")
            )
        # 4) fallback
        if dim is None:
            dim = 32

    assert dim is not None
    if dim <= 0:
        raise SystemExit(f"Invalid dim={dim}")

    # Filter sentences to vocab for an apples-to-apples comparison
    sentences_filt = filter_sentences_to_vocab(sentences, vocab_set)

    # Seed everything we control
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Word2Vec baseline training")
    print(f"  tokenized: {tokenized_path}")
    print(f"  tfidf:     {tfidf_path}")
    print(f"  vocab:     {len(vocab_order)} words (tf_idfs.keys())")
    print(f"  sentences: {len(sentences)} raw, {len(sentences_filt)} filtered (len>=2)")
    print(f"  dim:       {dim}")
    print(f"  window:    {args.window}")
    print(f"  sg:        {args.sg}  (1=skip-gram)")
    print(f"  negative:  {args.negative}")
    print(f"  epochs:    {args.epochs}")
    print(f"  workers:   {args.workers} (use 1 for determinism)")
    print(f"  sample:    {args.sample}")

    model = Word2Vec(
        sentences=sentences_filt,
        vector_size=dim,
        window=args.window,
        min_count=args.min_count,
        sg=args.sg,
        negative=args.negative,
        sample=args.sample,
        workers=args.workers,
        seed=args.seed,
        epochs=args.epochs,
    )

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("data/word2vec") / f"word2vec_vectors_{dim}d_window{args.window}.json"

    kept, missing = export_vectors_json(model, vocab_order, out_path)
    print(f"\nSaved: {out_path}")
    print(f"Exported vectors: {kept}  Missing from model: {missing}")

    if args.run_eval:
        # Run your evaluator
        import subprocess, sys as _sys
        cmd = [
            _sys.executable,
            "evaluation/evaluate_analogies.py",
            "--embeddings",
            str(out_path),
            "--questions",
            args.questions,
        ]
        print("\nRunning evaluator:")
        print("  " + " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()