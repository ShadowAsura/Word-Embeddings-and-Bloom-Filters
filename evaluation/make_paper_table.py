#!/usr/bin/env python3
"""
Generate the paper-ready analogy accuracy comparison table and bar charts.

Reads:
  - evaluation/window_sweep_results.csv          (diffusion, windows 2/4/6/8)
  - evaluation/word2vec_window_results.csv        (Word2Vec, windows 2/4/6/8)
  - data/nnlm/nnlm_eval_results.csv              (NNLM, if present)
  - data/rnnlm/rnnlm_eval_results.csv            (RNNLM, if present)

Writes:
  - results/paper_comparison_table.md
  - results/paper_comparison_table.tex
  - results/plots/paper_bar_chart.png
  - results/plots/paper_bar_chart_sem_syn.png

Run from project root:
  python evaluation/make_paper_table.py
  python evaluation/make_paper_table.py --diffusion-csv evaluation/diffusion_window_sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "results" / "plots"
OUT_MD = ROOT / "results" / "paper_comparison_table.md"
OUT_TEX = ROOT / "results" / "paper_comparison_table.tex"

WINDOWS = [2, 4, 6, 8]

# -------------------------------------------------------------------------
# Loaders
# -------------------------------------------------------------------------

def load_diffusion_window_sweep(path: Path) -> Dict[int, dict]:
    """
    window_sweep_results.csv columns: window, iter, semantic_acc, syntactic_acc, total_acc
    Returns best-total row per window.
    """
    if not path.is_file():
        return {}
    rows_by_window: Dict[int, list] = {w: [] for w in WINDOWS}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            w = int(r["window"])
            if w in rows_by_window:
                rows_by_window[w].append({
                    "iter": int(r["iter"]),
                    "semantic_acc": float(r["semantic_acc"]),
                    "syntactic_acc": float(r["syntactic_acc"]),
                    "total_acc": float(r["total_acc"]),
                })
    best: Dict[int, dict] = {}
    for w, rows in rows_by_window.items():
        if rows:
            best[w] = max(rows, key=lambda r: r["total_acc"])
    return best


def load_diffusion_sweep_full(path: Path) -> Dict[int, dict]:
    """
    diffusion_window_sweep_results.csv columns: window, iter, bits, semantic_acc, syntactic_acc, total_acc, ...
    Returns best-total row per window.
    """
    if not path.is_file():
        return {}
    rows_by_window: Dict[int, list] = {w: [] for w in WINDOWS}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            w = int(r["window"])
            if w in rows_by_window:
                rows_by_window[w].append({
                    "iter": int(r["iter"]),
                    "bits": int(r["bits"]),
                    "semantic_acc": float(r["semantic_acc"]),
                    "syntactic_acc": float(r["syntactic_acc"]),
                    "total_acc": float(r["total_acc"]),
                })
    best: Dict[int, dict] = {}
    for w, rows in rows_by_window.items():
        if rows:
            best[w] = max(rows, key=lambda r: r["total_acc"])
    return best


def load_word2vec(path: Path) -> Dict[int, dict]:
    """word2vec_window_results.csv. Returns row per window."""
    if not path.is_file():
        return {}
    result: Dict[int, dict] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            w = int(r["window"])
            result[w] = {
                "semantic_acc": float(r["semantic_acc"]),
                "syntactic_acc": float(r["syntactic_acc"]),
                "total_acc": float(r["total_acc"]),
            }
    return result


def load_single_method(path: Path) -> Optional[dict]:
    """
    Generic single-row CSV for NNLM / RNNLM results.
    Expected columns: semantic_acc, syntactic_acc, total_acc
    Falls back to reading a JSON eval output.
    """
    if not path.is_file():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    r = rows[0]
    return {
        "semantic_acc": float(r.get("semantic_acc", 0)),
        "syntactic_acc": float(r.get("syntactic_acc", 0)),
        "total_acc": float(r.get("total_acc", 0)),
    }


# -------------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------------

def fmt(val: Optional[float], bold: bool = False) -> str:
    if val is None:
        return "—"
    s = f"{val:.1f}"
    return f"**{s}**" if bold else s


def fmt_tex(val: Optional[float], bold: bool = False) -> str:
    if val is None:
        return "—"
    s = f"{val:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


# -------------------------------------------------------------------------
# Table builders
# -------------------------------------------------------------------------

def build_rows(
    diffusion: Dict[int, dict],
    w2v: Dict[int, dict],
    nnlm: Optional[dict],
    rnnlm: Optional[dict],
) -> List[dict]:
    """Build rows for the paper table. One row per (method, window) combination."""
    rows = []

    # Diffusion rows (one per window)
    for w in WINDOWS:
        d = diffusion.get(w)
        rows.append({
            "method": f"Diffusion (N={w})",
            "window": w,
            "iter": d["iter"] if d else None,
            "semantic_acc": d["semantic_acc"] if d else None,
            "syntactic_acc": d["syntactic_acc"] if d else None,
            "total_acc": d["total_acc"] if d else None,
        })

    # Word2Vec rows
    for w in WINDOWS:
        d = w2v.get(w)
        rows.append({
            "method": f"Word2Vec (W={w})",
            "window": w,
            "iter": None,
            "semantic_acc": d["semantic_acc"] if d else None,
            "syntactic_acc": d["syntactic_acc"] if d else None,
            "total_acc": d["total_acc"] if d else None,
        })

    # NNLM (single best row; window not applicable)
    if nnlm:
        rows.append({
            "method": "NNLM",
            "window": None,
            "iter": None,
            "semantic_acc": nnlm["semantic_acc"],
            "syntactic_acc": nnlm["syntactic_acc"],
            "total_acc": nnlm["total_acc"],
        })

    # RNNLM
    if rnnlm:
        rows.append({
            "method": "RNNLM",
            "window": None,
            "iter": None,
            "semantic_acc": rnnlm["semantic_acc"],
            "syntactic_acc": rnnlm["syntactic_acc"],
            "total_acc": rnnlm["total_acc"],
        })

    return rows


def write_markdown(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Find best total per category for bolding
    all_totals = [r["total_acc"] for r in rows if r["total_acc"] is not None]
    best_total = max(all_totals) if all_totals else None

    lines = [
        "# Analogy Accuracy Comparison Table",
        "",
        "All accuracies as percentages (%). Bold = best total.",
        "",
        "| Method | Semantic % | Syntactic % | Total % |",
        "|--------|-----------|------------|--------|",
    ]
    for r in rows:
        is_best = r["total_acc"] is not None and best_total is not None and abs(r["total_acc"] - best_total) < 1e-6
        lines.append(
            f"| {r['method']} | {fmt(r['semantic_acc'])} | {fmt(r['syntactic_acc'])} | {fmt(r['total_acc'], bold=is_best)} |"
        )
    lines += ["", f"_Generated by make_paper_table.py_", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def write_latex(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_totals = [r["total_acc"] for r in rows if r["total_acc"] is not None]
    best_total = max(all_totals) if all_totals else None

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Analogy accuracy comparison (\%). All methods trained on the same fairytale corpus.}",
        r"\label{tab:analogy_comparison}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Semantic (\%) & Syntactic (\%) & Total (\%) \\",
        r"\midrule",
    ]

    prev_family = None
    for r in rows:
        # Add a midrule between method families
        family = "diffusion" if "Diffusion" in r["method"] else \
                 "word2vec" if "Word2Vec" in r["method"] else r["method"]
        if prev_family is not None and family != prev_family:
            lines.append(r"\midrule")
        prev_family = family

        is_best = r["total_acc"] is not None and best_total is not None and abs(r["total_acc"] - best_total) < 1e-6
        lines.append(
            f"  {r['method']} & {fmt_tex(r['semantic_acc'])} & {fmt_tex(r['syntactic_acc'])} & {fmt_tex(r['total_acc'], bold=is_best)} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------

def plot_bar_chart(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["method"] for r in rows]
    totals = [r["total_acc"] if r["total_acc"] is not None else 0 for r in rows]
    colors = []
    for r in rows:
        if "Diffusion" in r["method"]:
            colors.append("#4C72B0")
        elif "Word2Vec" in r["method"]:
            colors.append("#DD8452")
        elif "NNLM" in r["method"]:
            colors.append("#55A868")
        else:
            colors.append("#C44E52")

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    bars = ax.bar(x, totals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Total Accuracy (%)", fontsize=11)
    ax.set_title("Analogy Total Accuracy: All Methods", fontsize=13)
    ax.set_ylim(0, max(totals) * 1.25 + 2)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Diffusion (Ours)"),
        Patch(facecolor="#DD8452", label="Word2Vec"),
    ]
    if any("NNLM" in r["method"] for r in rows):
        legend_elements.append(Patch(facecolor="#55A868", label="NNLM"))
    if any("RNNLM" in r["method"] for r in rows):
        legend_elements.append(Patch(facecolor="#C44E52", label="RNNLM"))
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_sem_syn_grouped(rows: List[dict], out_path: Path) -> None:
    """Grouped bar chart: semantic vs syntactic per method."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["method"] for r in rows]
    sem = [r["semantic_acc"] if r["semantic_acc"] is not None else 0 for r in rows]
    syn = [r["syntactic_acc"] if r["syntactic_acc"] is not None else 0 for r in rows]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 5))
    ax.bar(x - width / 2, sem, width, label="Semantic", color="#4C72B0", alpha=0.9)
    ax.bar(x + width / 2, syn, width, label="Syntactic", color="#DD8452", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Analogy Accuracy: Semantic vs Syntactic", fontsize=13)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--diffusion-csv",
        default=str(ROOT / "evaluation" / "window_sweep_results.csv"),
        help="Path to diffusion window sweep CSV (window_sweep_results.csv or diffusion_window_sweep_results.csv)",
    )
    ap.add_argument("--w2v-csv", default=str(ROOT / "evaluation" / "word2vec_window_results.csv"))
    ap.add_argument("--nnlm-csv", default=str(ROOT / "data" / "nnlm" / "nnlm_eval_results.csv"))
    ap.add_argument("--rnnlm-csv", default=str(ROOT / "data" / "rnnlm" / "rnnlm_eval_results.csv"))
    args = ap.parse_args()

    # Try window_sweep_results first, then diffusion_window_sweep
    diff_path = Path(args.diffusion_csv)
    if "window_sweep_results" in str(diff_path) and diff_path.is_file():
        diffusion = load_diffusion_window_sweep(diff_path)
    else:
        diffusion = load_diffusion_sweep_full(diff_path)

    w2v = load_word2vec(Path(args.w2v_csv))
    nnlm = load_single_method(Path(args.nnlm_csv))
    rnnlm = load_single_method(Path(args.rnnlm_csv))

    print(f"Loaded diffusion: {len(diffusion)} windows")
    print(f"Loaded Word2Vec:  {len(w2v)} windows")
    print(f"Loaded NNLM:  {'yes' if nnlm else 'no (CSV not found)'}")
    print(f"Loaded RNNLM: {'yes' if rnnlm else 'no (CSV not found)'}")

    rows = build_rows(diffusion, w2v, nnlm, rnnlm)

    write_markdown(rows, OUT_MD)
    write_latex(rows, OUT_TEX)
    plot_bar_chart(rows, PLOTS_DIR / "paper_bar_chart.png")
    plot_sem_syn_grouped(rows, PLOTS_DIR / "paper_bar_chart_sem_syn.png")

    # Pretty-print to terminal
    print("\n" + "=" * 65)
    print(f"{'Method':<22} {'Semantic':>10} {'Syntactic':>10} {'Total':>8}")
    print("-" * 55)
    for r in rows:
        sem = f"{r['semantic_acc']:.1f}%" if r["semantic_acc"] is not None else "—"
        syn = f"{r['syntactic_acc']:.1f}%" if r["syntactic_acc"] is not None else "—"
        tot = f"{r['total_acc']:.1f}%" if r["total_acc"] is not None else "—"
        print(f"  {r['method']:<20} {sem:>10} {syn:>10} {tot:>8}")
    print("=" * 65)


if __name__ == "__main__":
    main()
