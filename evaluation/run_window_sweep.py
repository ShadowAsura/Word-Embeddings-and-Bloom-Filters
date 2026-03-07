#!/usr/bin/env python3
"""
Window (NEIGHBORHOOD_SIZE) sweep: train with window in [2, 4, 6, 8],
ALPHA=0.1, USE_ROBUST_SCALING=0, ITERATIONS=50; evaluate at 1,5,10,25,50;
write window_sweep_results.csv and window_sweep_summary.txt.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = [2, 4, 6, 8]
ALPHA = 0.1
CHECKPOINTS = [1, 5, 10, 25, 50]
ITERATIONS = 50
VEC_DIR = os.path.join(ROOT, "data", "iterative_vectors")
EVAL_SCRIPT = os.path.join(ROOT, "evaluation", "evaluate_analogies.py")
TRAIN_SCRIPT = os.path.join(ROOT, "iterative_vectors_v3.py")
CSV_PATH = os.path.join(ROOT, "evaluation", "window_sweep_results.csv")
SUMMARY_PATH = os.path.join(ROOT, "evaluation", "window_sweep_summary.txt")


def run_training(window: int):
    env = os.environ.copy()
    env["NEIGHBORHOOD_SIZE"] = str(window)
    env["ALPHA"] = str(ALPHA)
    env["USE_ROBUST_SCALING"] = "0"
    env["ITERATIONS"] = str(ITERATIONS)
    subprocess.run(
        [sys.executable, TRAIN_SCRIPT],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=False,
    )


def get_embedding_path(window: int, iter_num: int):
    return os.path.join(
        VEC_DIR,
        f"window_{window}_iter_{iter_num}_v3_32bit_alpha{ALPHA}_norobust.json",
    )


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
    for window in WINDOWS:
        print(f"\n{'='*60}\nTraining window={window} (ALPHA={ALPHA}, no robust, iterations={ITERATIONS})\n{'='*60}")
        run_training(window)
        for iter_num in CHECKPOINTS:
            path = get_embedding_path(window, iter_num)
            print(f"  Evaluating {path} ...")
            res = run_evaluate(path)
            if res is not None:
                sem, syn, total = res
                rows.append({
                    "window": window,
                    "iter": iter_num,
                    "semantic_acc": sem,
                    "syntactic_acc": syn,
                    "total_acc": total,
                })
                print(f"    -> Semantic {sem:.1f}% Syntactic {syn:.1f}% Total {total:.1f}%")
            else:
                print(f"    -> (skip or failed)")

    # Write CSV
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["window", "iter", "semantic_acc", "syntactic_acc", "total_acc"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {CSV_PATH}")

    # Summary: per window best total, iter where total peaks
    summary_lines = [
        "WINDOW SWEEP SUMMARY (ALPHA=0.1, USE_ROBUST_SCALING=0)",
        "=" * 60,
        "",
        f"{'Window':<8} {'Best Iter':<10} {'Semantic':<10} {'Syntactic':<10} {'Total':<10}",
        "-" * 50,
    ]
    print("\n" + "\n".join(summary_lines))
    for window in WINDOWS:
        w_rows = [r for r in rows if r["window"] == window]
        if not w_rows:
            line = f"{window:<8} {'—':<10} {'—':<10} {'—':<10} {'—':<10}"
            summary_lines.append(line)
            print(line)
            continue
        best_total_row = max(w_rows, key=lambda r: r["total_acc"])
        best_iter = best_total_row["iter"]
        sem_at_best = best_total_row["semantic_acc"]
        syn_at_best = best_total_row["syntactic_acc"]
        total_at_best = best_total_row["total_acc"]
        line = f"{window:<8} {best_iter:<10} {sem_at_best:<10.1f} {syn_at_best:<10.1f} {total_at_best:<10.1f}"
        summary_lines.append(line)
        print(line)
    summary_lines.append("=" * 60)
    print("=" * 60)

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\nWrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
