#!/usr/bin/env python3
"""
Evaluate diffusion checkpoints and Word2Vec baseline embeddings; write CSV for paper table.
Usage: python evaluation/run_paper_eval.py
"""
from __future__ import annotations

import csv
import glob
import os
import re
import subprocess
import sys

import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SCRIPT = os.path.join(ROOT, "evaluation", "evaluate_analogies.py")
QUESTIONS = os.path.join(ROOT, "evaluation", "analogy_questions.json")
VEC_DIR = os.path.join(ROOT, "data", "iterative_vectors")
OUT_CSV = os.path.join(ROOT, "results", "analogies", "paper_analogy_results.csv")


def parse_eval_output(stdout: str):
    sem_acc = syn_acc = total_acc = None
    sem_valid = syn_valid = total_valid = None
    for line in stdout.splitlines():
        m = re.match(r"Semantic\s+(\d+)/(\d+)\s+([\d.]+)%", line)
        if m:
            sem_valid, _, sem_acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
        m = re.match(r"Syntactic\s+(\d+)/(\d+)\s+([\d.]+)%", line)
        if m:
            syn_valid, _, syn_acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
        m = re.match(r"Total\s+(\d+)/(\d+)\s+([\d.]+)%", line)
        if m:
            total_valid, _, total_acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return sem_acc, syn_acc, total_acc, sem_valid, syn_valid, total_valid


def run_evaluate(embedding_path: str, questions_path: str):
    if not os.path.isfile(embedding_path):
        return None
    result = subprocess.run(
        [sys.executable, EVAL_SCRIPT, "--embeddings", embedding_path, "--questions", questions_path],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return parse_eval_output(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Evaluate diffusion + Word2Vec embeddings for analogy table.")
    parser.add_argument(
        "--questions",
        default=QUESTIONS,
        help="Path to analogy questions JSON (default: evaluation/analogy_questions.json)",
    )
    args = parser.parse_args()

    questions_path = args.questions

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    rows = []

    # Diffusion checkpoints: window_N_iter_*_v3_*bit.json
    for window in [2, 4, 6, 8]:
        pattern = os.path.join(VEC_DIR, f"window_{window}_iter_*_v3_*bit.json")
        files = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"_iter_(\d+)_", p).group(1)) if re.search(r"_iter_(\d+)_", p) else 0)
        for path in files:
            m = re.search(r"window_(\d+)_iter_(\d+)_", path)
            if not m:
                continue
            w, it = int(m.group(1)), int(m.group(2))
            res = run_evaluate(path, questions_path)
            if res is None:
                continue
            sem_acc, syn_acc, total_acc, sem_valid, syn_valid, total_valid = res
            rows.append({
                "method": "diffusion",
                "window": w,
                "iter": it,
                "semantic_acc": sem_acc,
                "syntactic_acc": syn_acc,
                "total_acc": total_acc,
                "semantic_valid": sem_valid or "",
                "syntactic_valid": syn_valid or "",
                "total_valid": total_valid or "",
                "embedding_path": path,
            })

    # Word2Vec baselines
    for window in [2, 4, 6, 8]:
        path = os.path.join(ROOT, "data", "word2vec", f"word2vec_vectors_32d_window{window}.json")
        if not os.path.isfile(path):
            continue
        res = run_evaluate(path, questions_path)
        if res is None:
            continue
        sem_acc, syn_acc, total_acc, sem_valid, syn_valid, total_valid = res
        rows.append({
            "method": "word2vec",
            "window": window,
            "iter": -1,
            "semantic_acc": sem_acc,
            "syntactic_acc": syn_acc,
            "total_acc": total_acc,
            "semantic_valid": sem_valid or "",
            "syntactic_valid": syn_valid or "",
            "total_valid": total_valid or "",
            "embedding_path": path,
        })

    # Consistent row order: Diffusion N=2,4,6,8 (by window, then iter), then Word2Vec window=2,4,6,8
    def row_key(r):
        method_order = 0 if r["method"] == "diffusion" else 1
        return (method_order, int(r["window"]), int(r["iter"]))

    rows = sorted(rows, key=row_key)

    # One decimal place for accuracies; keep valid counts as-is
    for r in rows:
        for k in ("semantic_acc", "syntactic_acc", "total_acc"):
            if r.get(k) is not None:
                r[k] = round(float(r[k]), 1)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method", "window", "iter", "semantic_acc", "syntactic_acc", "total_acc",
                "semantic_valid", "syntactic_valid", "total_valid", "embedding_path",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
