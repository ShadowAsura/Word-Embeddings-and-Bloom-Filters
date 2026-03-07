from __future__ import annotations

import argparse
import json
import os
from typing import Iterable

import numpy as np


DEFAULT_CHECKPOINTS = [0, 1, 2, 10, 50, 100, 200, 399]


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def compare_iteration(iteration: int, generated_path: str, ground_truth_path: str) -> None:
    generated = load_json(generated_path)
    ground_truth = load_json(ground_truth_path)

    gen_vocab = set(generated.keys())
    gt_vocab = set(ground_truth.keys())
    shared_vocab = sorted(gen_vocab & gt_vocab)

    if not shared_vocab:
        raise ValueError(f"No shared vocab for iteration {iteration}")

    cosine_scores = []
    max_abs_diff = 0.0

    for word in shared_vocab:
        gen_vec = np.asarray(generated[word], dtype=np.float64)
        gt_vec = np.asarray(ground_truth[word], dtype=np.float64)
        cosine_scores.append(cosine_similarity(gen_vec, gt_vec))
        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(gen_vec - gt_vec))))

    cosine_scores = np.asarray(cosine_scores, dtype=np.float64)

    print(f"Iteration {iteration}")
    print(f"  generated file: {generated_path}")
    print(f"  ground truth  : {ground_truth_path}")
    print(f"  vocab size generated: {len(gen_vocab)}")
    print(f"  vocab size truth    : {len(gt_vocab)}")
    print(f"  vocab sizes match   : {len(gen_vocab) == len(gt_vocab)}")
    print(f"  shared vocab        : {len(shared_vocab)}")
    print(f"  generated-only vocab: {len(gen_vocab - gt_vocab)}")
    print(f"  truth-only vocab    : {len(gt_vocab - gen_vocab)}")
    print(
        "  cosine min/mean/max: "
        f"{cosine_scores.min():.6f} / {cosine_scores.mean():.6f} / {cosine_scores.max():.6f}"
    )
    print(f"  max abs diff        : {max_abs_diff:.6e}")


def parse_iterations(raw: str) -> Iterable[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generated-dir",
        default=os.path.join("data", "iterative_vectors"),
        help="Directory containing generated GPU outputs",
    )
    parser.add_argument(
        "--generated-pattern",
        default="window_4_iter_{iteration}_cpu_equiv_32bit.json",
        help="Filename pattern for generated outputs",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=os.path.join("data", "iterative_vectors"),
        help="Directory containing historical CPU outputs",
    )
    parser.add_argument(
        "--ground-truth-pattern",
        default="{iteration}.json",
        help="Filename pattern for historical CPU outputs",
    )
    parser.add_argument(
        "--iterations",
        default=",".join(str(i) for i in DEFAULT_CHECKPOINTS),
        help="Comma-separated iterations to compare",
    )
    args = parser.parse_args()

    iterations = parse_iterations(args.iterations)
    for iteration in iterations:
        generated_path = os.path.join(
            args.generated_dir, args.generated_pattern.format(iteration=iteration)
        )
        ground_truth_path = os.path.join(
            args.ground_truth_dir, args.ground_truth_pattern.format(iteration=iteration)
        )
        if not os.path.exists(generated_path):
            raise FileNotFoundError(f"Generated file not found: {generated_path}")
        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"Ground-truth file not found: {ground_truth_path}")
        compare_iteration(iteration, generated_path, ground_truth_path)


if __name__ == "__main__":
    main()
