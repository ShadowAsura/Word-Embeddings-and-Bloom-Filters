#!/usr/bin/env python3
"""
Evaluate Word2Vec embeddings (windows 2,4,6,8) with evaluate_analogies.py;
write evaluation/word2vec_window_results.csv.
"""
from __future__ import annotations

import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = [2, 4, 6, 8]
EVAL_SCRIPT = os.path.join(ROOT, "evaluation", "evaluate_analogies.py")
QUESTIONS = os.path.join(ROOT, "evaluation", "analogy_questions.json")
OUT_CSV = os.path.join(ROOT, "evaluation", "word2vec_window_results.csv")


def parse_eval_output(stdout: str):
    sem_acc = syn_acc = total_acc = None
    sem_valid = syn_valid = total_valid = None
    for line in stdout.splitlines():
        # Supports both formats:
        # - old:  "Semantic      20      20     16.7%"
        # - new:  "Semantic      20/20       16.7%"
        m = re.match(r"Semantic\s+(\d+)(?:/|\s+)(\d+)\s+([\d.]+)%", line)
        if m:
            sem_valid, _, sem_acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
        m = re.match(r"Syntactic\s+(\d+)(?:/|\s+)(\d+)\s+([\d.]+)%", line)
        if m:
            syn_valid, _, syn_acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
        m = re.match(r"Total\s+(\d+)(?:/|\s+)(\d+)\s+([\d.]+)%", line)
        if m:
            total_valid, _, total_acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return sem_acc, syn_acc, total_acc, total_valid


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    rows = []
    for window in WINDOWS:
        path = os.path.join(ROOT, "data", "word2vec", f"word2vec_vectors_32d_window{window}.json")
        if not os.path.isfile(path):
            print(f"Skip (missing): {path}")
            continue
        result = subprocess.run(
            [sys.executable, EVAL_SCRIPT, "--embeddings", path, "--questions", QUESTIONS],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Eval failed: {path}")
            continue
        sem_acc, syn_acc, total_acc, total_valid = parse_eval_output(result.stdout)
        rows.append({
            "method": "word2vec",
            "window": window,
            "iter": -1,
            "semantic_acc": sem_acc,
            "syntactic_acc": syn_acc,
            "total_acc": total_acc,
            "total_valid": total_valid or "",
            "embedding_path": path,
        })
        print(f"window={window} total_acc={total_acc:.1f}%")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["method", "window", "iter", "semantic_acc", "syntactic_acc", "total_acc", "total_valid", "embedding_path"],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
