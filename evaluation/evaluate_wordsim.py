#!/usr/bin/env python3
"""
Evaluate embedding JSON files on WordSim-353 using Spearman rank correlation.

Input embedding format:
  {"word": [float, float, ...], ...}

Output format:
  Pairs: VALID/TOTAL
  Spearman rho: 0.XXX (p=X.XXe-XX)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr


def load_embeddings(path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = list(data.keys())
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    mat = np.asarray([data[w] for w in vocab], dtype=np.float32)

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mat = mat / norms
    return mat, word_to_idx


def load_wordsim_pairs(path: Path) -> List[Tuple[str, str, float]]:
    pairs: List[Tuple[str, str, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # WordSim-353 is usually tab-separated with optional header.
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
                if len(parts) < 3:
                    continue

            w1, w2 = parts[0].strip(), parts[1].strip()
            try:
                score = float(parts[2])
            except ValueError:
                # Skip header or malformed lines.
                continue

            pairs.append((w1, w2, score))

    return pairs


def evaluate_wordsim(
    vectors: np.ndarray,
    word_to_idx: Dict[str, int],
    pairs: List[Tuple[str, str, float]],
) -> Tuple[int, int, float, float]:
    total = len(pairs)
    model_scores: List[float] = []
    human_scores: List[float] = []

    for w1, w2, human_score in pairs:
        if w1 not in word_to_idx or w2 not in word_to_idx:
            continue

        i = word_to_idx[w1]
        j = word_to_idx[w2]
        cosine_sim = float(np.dot(vectors[i], vectors[j]))

        model_scores.append(cosine_sim)
        human_scores.append(human_score)

    valid = len(model_scores)
    if valid < 2:
        return valid, total, 0.0, 1.0

    rho, p_value = spearmanr(model_scores, human_scores)
    if rho is None or np.isnan(rho):
        rho = 0.0
    if p_value is None or np.isnan(p_value):
        p_value = 1.0

    return valid, total, float(rho), float(p_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embeddings on WordSim-353")
    parser.add_argument("--embeddings", required=True, help="Path to embedding JSON")
    parser.add_argument("--wordsim", required=True, help="Path to wordsim353.txt")
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    wordsim_path = Path(args.wordsim)

    if not emb_path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not wordsim_path.is_file():
        raise FileNotFoundError(f"WordSim file not found: {wordsim_path}")

    vectors, word_to_idx = load_embeddings(emb_path)
    pairs = load_wordsim_pairs(wordsim_path)
    valid, total, rho, p_value = evaluate_wordsim(vectors, word_to_idx, pairs)

    print(f"Pairs: {valid}/{total}")
    print(f"Spearman rho: {rho:.3f} (p={p_value:.2e})")


if __name__ == "__main__":
    main()
