#!/usr/bin/env python3
"""
Reusable analogy evaluation (Word2Vec-style) for any embedding JSON file.
Usage: python evaluate_analogies.py --embeddings path/to/embeddings.json [--questions path/to/analogy_questions.json] [--show-wrong]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def load_embeddings(path: str):
    """Load embeddings from JSON. Returns V (float32, row-normalized), word_to_idx, vocab_list."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab_list = list(data.keys())
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}
    V = np.array([data[w] for w in vocab_list], dtype=np.float32)
    # Normalize rows to unit length
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = V / norms
    return V, word_to_idx, vocab_list


def load_questions(path: str):
    """Load analogy_questions.json. Returns dict with 'semantic' and 'syntactic' lists of [a, b, c, expected]."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def solve_analogy(V: np.ndarray, word_to_idx: dict, a: str, b: str, c: str, exclude: set):
    """
    Compute v = V[b] - V[a] + V[c], normalize v, then prediction = argmax(V @ v) excluding a, b, c.
    Returns (prediction_idx, scores for debugging). prediction_idx is -1 if any of a,b,c missing.
    """
    for w in (a, b, c):
        if w not in word_to_idx:
            return None, None
    ia, ib, ic = word_to_idx[a], word_to_idx[b], word_to_idx[c]
    v = V[ib] - V[ia] + V[ic]
    n = np.linalg.norm(v)
    if n == 0:
        return None, None
    v = v / n
    scores = V @ v
    exclude_idx = {ia, ib, ic}
    scores[list(exclude_idx)] = -np.inf
    prediction_idx = int(np.argmax(scores))
    return prediction_idx, scores


def evaluate_category(V: np.ndarray, word_to_idx: dict, idx_to_word: list, questions: list, category_name: str, show_wrong: bool):
    """Evaluate one category. Returns (valid, total, correct, wrong_list)."""
    total = len(questions)
    valid = 0
    correct = 0
    wrong_list = []
    for q in questions:
        if len(q) != 4:
            continue
        a, b, c, expected = q[0], q[1], q[2], q[3]
        if any(w not in word_to_idx for w in [a, b, c, expected]):
            continue
        valid += 1
        pred_idx, _ = solve_analogy(V, word_to_idx, a, b, c, set())
        if pred_idx is None:
            continue
        pred_word = idx_to_word[pred_idx]
        if pred_word == expected:
            correct += 1
        else:
            wrong_list.append((a, b, c, expected, pred_word))
    if show_wrong and wrong_list:
        print(f"\n  Incorrect {category_name}:")
        for a, b, c, exp, pred in wrong_list[:20]:
            print(f"    {a} - {b} + {c} = {pred} (expected {exp})")
        if len(wrong_list) > 20:
            print(f"    ... and {len(wrong_list) - 20} more")
    return valid, total, correct, wrong_list


def main():
    parser = argparse.ArgumentParser(description="Evaluate embeddings on analogy questions")
    parser.add_argument("--embeddings", required=True, help="Path to embedding JSON file")
    parser.add_argument(
        "--questions",
        default=os.path.join(os.path.dirname(__file__), "analogy_questions.json"),
        help="Path to analogy_questions.json",
    )
    parser.add_argument("--show-wrong", action="store_true", help="Print incorrect analogies")
    args = parser.parse_args()

    if not os.path.isfile(args.embeddings):
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings}")
    if not os.path.isfile(args.questions):
        raise FileNotFoundError(f"Questions file not found: {args.questions}")

    V, word_to_idx, vocab_list = load_embeddings(args.embeddings)
    idx_to_word = vocab_list
    questions_data = load_questions(args.questions)

    results = []
    total_valid = 0
    total_count = 0
    total_correct = 0

    for category_name in ("semantic", "syntactic"):
        if category_name not in questions_data:
            continue
        q_list = questions_data[category_name]
        valid, total, correct, _ = evaluate_category(
            V, word_to_idx, idx_to_word, q_list, category_name.capitalize(), args.show_wrong
        )
        results.append((category_name.capitalize(), valid, total, correct))
        total_valid += valid
        total_count += total
        total_correct += correct

    if total_valid > 0:
        overall_acc = 100.0 * total_correct / total_valid
    else:
        overall_acc = 0.0

    print(f"Embedding: {args.embeddings}")
    print(f"Category     Valid/Total   Accuracy")
    print("-" * 36)
    for cat, valid, total, correct in results:
        acc = (100.0 * correct / valid) if valid > 0 else 0.0
        print(f"{cat:<12} {valid:>3}/{total:<3}   {acc:>6.1f}%")
    print("-" * 36)
    print(f"{'Total':<12} {total_valid:>3}/{total_count:<3}   {overall_acc:>6.1f}%")


if __name__ == "__main__":
    main()
