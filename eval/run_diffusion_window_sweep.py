#!/usr/bin/env python3
"""
Run diffusion (iterative_vectors_v3.py) for windows 2,4,6,8 with ITERATIONS=150;
evaluate each checkpoint with evaluate_analogies.py; write eval/diffusion_window_sweep_results.csv.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = [2, 4, 6, 8]
ITERATIONS = 150
DIFFUSION_SCRIPT = os.path.join(ROOT, "iterative_vectors_v3.py")
EVAL_SCRIPT = os.path.join(ROOT, "evaluation", "evaluate_analogies.py")
QUESTIONS = os.path.join(ROOT, "evaluation", "analogy_questions.json")
VEC_DIR = os.path.join(ROOT, "data", "iterative_vectors")
OUT_CSV = os.path.join(ROOT, "eval", "diffusion_window_sweep_results.csv")


def run_diffusion(window: int):
    env = os.environ.copy()
    env["NEIGHBORHOOD_SIZE"] = str(window)
    env["ITERATIONS"] = str(ITERATIONS)
    subprocess.run(
        [sys.executable, DIFFUSION_SCRIPT],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=False,
    )


def parse_eval_output(stdout: str):
    """Parse evaluator stdout. Returns semantic_acc, syntactic_acc, total_acc, semantic_valid, syntactic_valid, total_valid."""
    sem_acc = syn_acc = total_acc = None
    sem_valid = syn_valid = total_valid = None
    for line in stdout.splitlines():
        m = re.match(r"Semantic\s+(\d+)\s+(\d+)\s+([\d.]+)%", line)
        if m:
            sem_valid, _, sem_acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
        m = re.match(r"Syntactic\s+(\d+)\s+(\d+)\s+([\d.]+)%", line)
        if m:
            syn_valid, _, syn_acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
        m = re.match(r"Total\s+(\d+)\s+(\d+)\s+([\d.]+)%", line)
        if m:
            total_valid, _, total_acc = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return sem_acc, syn_acc, total_acc, sem_valid, syn_valid, total_valid


def run_evaluate(embedding_path: str):
    if not os.path.isfile(embedding_path):
        return None
    result = subprocess.run(
        [sys.executable, EVAL_SCRIPT, "--embeddings", embedding_path, "--questions", QUESTIONS],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return parse_eval_output(result.stdout)


def parse_diffusion_filename(path: str):
    """Extract window, iter, bits from path like .../window_4_iter_9_v3_32bit.json."""
    basename = os.path.basename(path)
    m = re.match(r"window_(\d+)_iter_(\d+)_v3_(\d+)bit\.json", basename)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    rows = []
    for window in WINDOWS:
        print(f"\n{'='*60}\nDiffusion window={window} ITERATIONS={ITERATIONS}\n{'='*60}")
        run_diffusion(window)
        pattern = os.path.join(VEC_DIR, f"window_{window}_iter_*_v3_*bit.json")
        files = sorted(glob.glob(pattern), key=lambda p: parse_diffusion_filename(p) or (0, 0, 0))
        for path in files:
            info = parse_diffusion_filename(path)
            if not info:
                continue
            w, it, bits = info
            print(f"  Evaluate {path} ...")
            res = run_evaluate(path)
            if res is None:
                continue
            sem_acc, syn_acc, total_acc, sem_valid, syn_valid, total_valid = res
            rows.append({
                "window": w,
                "iter": it,
                "bits": bits,
                "semantic_acc": sem_acc,
                "syntactic_acc": syn_acc,
                "total_acc": total_acc,
                "semantic_valid": sem_valid or "",
                "syntactic_valid": syn_valid or "",
                "total_valid": total_valid or "",
                "embedding_path": path,
            })
            print(f"    total_acc={total_acc:.1f}%")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "window", "iter", "bits", "semantic_acc", "syntactic_acc", "total_acc",
                "semantic_valid", "syntactic_valid", "total_valid", "embedding_path",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
