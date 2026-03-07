#!/usr/bin/env python3
"""
Run controlled diffusion sweep: train with ALPHA in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
ITERATIONS=50, NEIGHBORHOOD_SIZE=4; save checkpoints 1,5,10,25,50; evaluate analogies; write CSV.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
CHECKPOINTS = [1, 5, 10, 25, 50]
WINDOW = 4
ITERATIONS = 50
VEC_DIR = os.path.join(ROOT, "data", "iterative_vectors")
EVAL_SCRIPT = os.path.join(ROOT, "evaluation", "evaluate_analogies.py")
TRAIN_SCRIPT = os.path.join(ROOT, "iterative_vectors_v3.py")
CSV_PATH = os.path.join(ROOT, "evaluation", "alpha_sweep_results.csv")


def run_training(alpha: float):
    env = os.environ.copy()
    env["ALPHA"] = str(alpha)
    env["NEIGHBORHOOD_SIZE"] = str(WINDOW)
    env["ITERATIONS"] = str(ITERATIONS)
    subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=False,
    )


def get_embedding_path(alpha: float, iter_num: int):
    return os.path.join(VEC_DIR, f"window_{WINDOW}_iter_{iter_num}_v3_32bit.json")


def run_evaluate(embedding_path: str) -> tuple[float, float, float] | None:
    if not os.path.isfile(embedding_path):
        return None
    result = subprocess.run(
        [sys.executable, EVAL_SCRIPT, "--embeddings", embedding_path],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    sem_acc = syn_acc = total_acc = None
    for line in result.stdout.splitlines():
        m = re.match(r"Semantic\s+\d+\s+\d+\s+([\d.]+)%", line)
        if m:
            sem_acc = float(m.group(1))
        m = re.match(r"Syntactic\s+\d+\s+\d+\s+([\d.]+)%", line)
        if m:
            syn_acc = float(m.group(1))
        m = re.match(r"Total\s+\d+\s+\d+\s+([\d.]+)%", line)
        if m:
            total_acc = float(m.group(1))
    if sem_acc is not None and syn_acc is not None and total_acc is not None:
        return (sem_acc, syn_acc, total_acc)
    return None


def main():
    os.makedirs(VEC_DIR, exist_ok=True)
    rows = []
    for alpha in ALPHAS:
        print(f"\n{'='*60}\nTraining ALPHA={alpha} (iterations={ITERATIONS}, window={WINDOW})\n{'='*60}")
        run_training(alpha)
        for iter_num in CHECKPOINTS:
            path = get_embedding_path(alpha, iter_num)
            print(f"  Evaluating {path} ...")
            res = run_evaluate(path)
            if res is not None:
                sem, syn, total = res
                rows.append({
                    "window": WINDOW,
                    "alpha": alpha,
                    "iter": iter_num,
                    "semantic_acc": sem,
                    "syntactic_acc": syn,
                    "total_acc": total,
                })
                print(f"    -> Semantic {sem:.1f}% Syntactic {syn:.1f}% Total {total:.1f}%")
            else:
                print(f"    -> (skip or failed)")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["window", "alpha", "iter", "semantic_acc", "syntactic_acc", "total_acc"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {CSV_PATH}")


if __name__ == "__main__":
    main()
