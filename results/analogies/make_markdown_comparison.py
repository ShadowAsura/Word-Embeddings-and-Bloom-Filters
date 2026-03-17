#!/usr/bin/env python3
"""
Build comparison table from results/analogies/paper_analogy_results.csv.
"""

from __future__ import annotations

import csv
import json
import os
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "analogies" / "paper_analogy_results.csv"
DEFAULT_QUESTIONS_PATH = ROOT / "evaluation" / "analogy_questions_shared.json"
OUT_MD = ROOT / "results" / "analogies" / "comparison_table.md"


def load_counts(questions_path: Path) -> tuple[int, int, int]:
    with questions_path.open("r", encoding="utf-8") as f:
        q = json.load(f)
    sem_total = len(q.get("semantic", []))
    syn_total = len(q.get("syntactic", []))
    total_total = sem_total + syn_total
    return sem_total, syn_total, total_total


def load_rows():
    rows = []
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            # Cast numeric fields
            r["window"] = int(r["window"])
            r["iter"] = int(r["iter"])
            r["semantic_acc"] = float(r["semantic_acc"])
            r["syntactic_acc"] = float(r["syntactic_acc"])
            r["total_acc"] = float(r["total_acc"])
            r["semantic_valid"] = int(r["semantic_valid"]) if r["semantic_valid"] != "" else 0
            r["syntactic_valid"] = int(r["syntactic_valid"]) if r["syntactic_valid"] != "" else 0
            r["total_valid"] = int(r["total_valid"]) if r["total_valid"] != "" else 0
            rows.append(r)
    return rows


def best_diffusion_by_window(rows):
    best = {}
    for r in rows:
        if r["method"] != "diffusion":
            continue
        w = r["window"]
        if w not in best or r["total_acc"] > best[w]["total_acc"]:
            best[w] = r
    return best


def main():
    ap = argparse.ArgumentParser(description="Build comparison table from paper_analogy_results.csv.")
    ap.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTIONS_PATH),
        help="Analogy questions JSON used to compute denominators (default: shared questions).",
    )
    args = ap.parse_args()

    questions_path = Path(args.questions)
    sem_total, syn_total, total_total = load_counts(questions_path)
    rows = load_rows()
    best_diff = best_diffusion_by_window(rows)

    # Helper to fetch Word2Vec row for a window
    def get_w2v(window: int):
        for r in rows:
            if r["method"] == "word2vec" and r["window"] == window:
                return r
        return None

    lines: list[str] = []
    lines.append("# Diffusion vs Word2Vec analogy summary")
    lines.append("")
    lines.append("| Method    | Config        | Best Iter | Semantic | Syntactic | Total | Semantic Valid/Total | Syntactic Valid/Total | Total Valid/Total |")
    lines.append("|-----------|--------------|-----------|----------|-----------|-------|----------------------|-----------------------|-------------------|")

    def add_row(method_label: str, config: str, r: dict | None):
        if r is None:
            return
        sem = f"{r['semantic_acc']:.1f}%"
        syn = f"{r['syntactic_acc']:.1f}%"
        tot = f"{r['total_acc']:.1f}%"
        sv = r["semantic_valid"]
        syv = r["syntactic_valid"]
        tv = r["total_valid"]
        sem_v = f"{sv}/{sem_total}"
        syn_v = f"{syv}/{syn_total}"
        tot_v = f"{tv}/{total_total}"
        best_iter = r["iter"] if r["iter"] >= 0 else "—"
        lines.append(
            f"| {method_label:<9} | {config:<12} | {best_iter!s:<9} | {sem:<8} | {syn:<9} | {tot:<5} | {sem_v:<20} | {syn_v:<21} | {tot_v:<17} |"
        )

    # Diffusion N=2,4,6,8
    for n in (2, 4, 6, 8):
        add_row("Diffusion", f"N={n}", best_diff.get(n))

    # Word2Vec window=2,4,6,8
    for w in (2, 4, 6, 8):
        add_row("Word2Vec", f"window={w}", get_w2v(w))

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()

