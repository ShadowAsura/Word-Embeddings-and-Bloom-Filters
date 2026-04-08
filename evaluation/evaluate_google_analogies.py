#!/usr/bin/env python3
"""
Evaluate embedding JSONs on Google's official questions-words.txt benchmark.

Input embeddings format:
  {"word": [float, float, ...], ...}

Output format:
  Semantic   VALID/TOTAL   XX.X%
  Syntactic  VALID/TOTAL   XX.X%
  Total      VALID/TOTAL   XX.X%
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_embeddings(path: Path) -> Tuple[np.ndarray, Dict[str, int]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    stripped = raw.lstrip("\ufeff\t\r\n ")
    if not stripped:
        raise ValueError(f"Embeddings file is empty: {path}")
    if stripped.startswith("version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            "Embeddings file is a Git LFS pointer, not actual JSON vectors. "
            "Run `git lfs pull` or regenerate the embeddings file."
        )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid embeddings JSON in {path}: {exc}") from exc

    if not isinstance(data, dict) or not data:
        raise ValueError(f"Embeddings JSON must be a non-empty object: {path}")

    vocab = list(data.keys())
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    try:
        mat = np.asarray([data[w] for w in vocab], dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Embeddings contain non-numeric vectors in {path}: {exc}") from exc

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    mat = mat / norms
    return mat, word_to_idx


def load_google_questions(path: Path) -> Dict[str, List[Tuple[str, str, str, str]]]:
    grouped = {"semantic": [], "syntactic": []}
    current_group = "semantic"

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(":"):
                category = line[1:].strip().lower()
                current_group = "syntactic" if "gram" in category else "semantic"
                continue

            parts = line.split()
            if len(parts) != 4:
                continue
            a, b, c, d = parts
            grouped[current_group].append((a, b, c, d))

    return grouped


def evaluate_category(
    vectors: np.ndarray,
    word_to_idx: Dict[str, int],
    questions: List[Tuple[str, str, str, str]],
) -> Tuple[int, int, int]:
    total = len(questions)
    valid = 0
    correct = 0

    for a, b, c, d in questions:
        if a not in word_to_idx or b not in word_to_idx or c not in word_to_idx or d not in word_to_idx:
            continue

        ia = word_to_idx[a]
        ib = word_to_idx[b]
        ic = word_to_idx[c]
        id_expected = word_to_idx[d]

        valid += 1

        query = vectors[ib] - vectors[ia] + vectors[ic]
        qn = np.linalg.norm(query)
        if qn == 0.0:
            continue
        query = query / qn

        scores = vectors @ query
        scores[ia] = -np.inf
        scores[ib] = -np.inf
        scores[ic] = -np.inf

        pred = int(np.argmax(scores))
        if pred == id_expected:
            correct += 1

    return valid, total, correct


def format_line(name: str, valid: int, total: int, correct: int) -> str:
    acc = (100.0 * correct / valid) if valid > 0 else 0.0
    return f"{name:<10} {valid}/{total}   {acc:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embeddings on questions-words.txt")
    parser.add_argument("--embeddings", required=True, help="Path to embedding JSON")
    parser.add_argument("--questions", required=True, help="Path to questions-words.txt")
    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    q_path = Path(args.questions)
    if not emb_path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {emb_path}")
    if not q_path.is_file():
        raise FileNotFoundError(f"Questions file not found: {q_path}")

    try:
        vectors, word_to_idx = load_embeddings(emb_path)
        grouped = load_google_questions(q_path)
    except ValueError as exc:
        raise SystemExit(f"Input error: {exc}") from exc

    sem_valid, sem_total, sem_correct = evaluate_category(vectors, word_to_idx, grouped["semantic"])
    syn_valid, syn_total, syn_correct = evaluate_category(vectors, word_to_idx, grouped["syntactic"])

    total_valid = sem_valid + syn_valid
    total_total = sem_total + syn_total
    total_correct = sem_correct + syn_correct

    print(format_line("Semantic", sem_valid, sem_total, sem_correct))
    print(format_line("Syntactic", syn_valid, syn_total, syn_correct))
    print(format_line("Total", total_valid, total_total, total_correct))


if __name__ == "__main__":
    main()
