#!/usr/bin/env python3
"""
Read diffusion_window_sweep_results.csv and word2vec_window_results.csv;
produce comparison plots and print summary table.
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIFFUSION_CSV = os.path.join(ROOT, "evaluation", "diffusion_window_sweep_results.csv")
W2V_CSV = os.path.join(ROOT, "evaluation", "word2vec_window_results.csv")
PLOTS_DIR = os.path.join(ROOT, "results", "plots")
WINDOWS = [2, 4, 6, 8]


def load_diffusion():
    rows = []
    with open(DIFFUSION_CSV, newline="") as f:
        for r in csv.DictReader(f):
            r["window"] = int(r["window"])
            r["iter"] = int(r["iter"])
            r["bits"] = int(r["bits"])
            r["semantic_acc"] = float(r["semantic_acc"])
            r["syntactic_acc"] = float(r["syntactic_acc"])
            r["total_acc"] = float(r["total_acc"])
            rows.append(r)
    return rows


def load_word2vec():
    rows = []
    with open(W2V_CSV, newline="") as f:
        for r in csv.DictReader(f):
            r["window"] = int(r["window"])
            r["iter"] = int(r["iter"])
            r["semantic_acc"] = float(r["semantic_acc"])
            r["syntactic_acc"] = float(r["syntactic_acc"])
            r["total_acc"] = float(r["total_acc"])
            rows.append(r)
    return rows


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    diff = load_diffusion()
    w2v = load_word2vec()
    w2v_by_window = {r["window"]: r for r in w2v}

    # --- 1) Diffusion total accuracy vs iter (one curve per window)
    fig, ax = plt.subplots()
    for window in WINDOWS:
        sub = [r for r in diff if r["window"] == window]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: r["iter"])
        ax.plot([r["iter"] for r in sub], [r["total_acc"] for r in sub], marker="o", label=f"window={window}")
    ax.set_xlabel("iter")
    ax.set_ylabel("Total accuracy (%)")
    ax.set_title("Diffusion: total accuracy vs iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(PLOTS_DIR, "diffusion_total_vs_iter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 2) Diffusion semantic vs syntactic vs iter
    fig, ax = plt.subplots()
    for window in WINDOWS:
        sub = [r for r in diff if r["window"] == window]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: r["iter"])
        iters = [r["iter"] for r in sub]
        ax.plot(iters, [r["semantic_acc"] for r in sub], marker="s", linestyle="-", label=f"semantic W={window}")
        ax.plot(iters, [r["syntactic_acc"] for r in sub], marker="^", linestyle="--", label=f"syntactic W={window}")
    ax.set_xlabel("iter")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Diffusion: semantic & syntactic vs iteration")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(PLOTS_DIR, "diffusion_sem_syn_vs_iter.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Best diffusion per window (by total_acc)
    best_diff_total = {}
    best_diff_semantic = {}
    for window in WINDOWS:
        sub = [r for r in diff if r["window"] == window]
        if sub:
            by_total = max(sub, key=lambda r: r["total_acc"])
            best_diff_total[window] = (by_total["iter"], by_total["total_acc"])
            by_sem = max(sub, key=lambda r: r["semantic_acc"])
            best_diff_semantic[window] = (by_sem["iter"], by_sem["semantic_acc"])
        else:
            best_diff_total[window] = (None, None)
            best_diff_semantic[window] = (None, None)

    # --- 3) Bar chart: best diffusion total vs Word2Vec total by window
    fig, ax = plt.subplots()
    x = range(len(WINDOWS))
    width = 0.35
    diff_vals = [best_diff_total[w][1] if best_diff_total[w][1] is not None else 0 for w in WINDOWS]
    w2v_vals = [w2v_by_window[w]["total_acc"] if w in w2v_by_window else 0 for w in WINDOWS]
    ax.bar([i - width / 2 for i in x], diff_vals, width, label="diffusion_best")
    ax.bar([i + width / 2 for i in x], w2v_vals, width, label="word2vec")
    ax.set_xticks(x)
    ax.set_xticklabels([f"W={w}" for w in WINDOWS])
    ax.set_ylabel("Total accuracy (%)")
    ax.set_title("Best total accuracy by window: Word2Vec vs Diffusion")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(PLOTS_DIR, "best_total_by_window_word2vec_vs_diffusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 4) Bar chart: best diffusion semantic vs Word2Vec semantic by window
    fig, ax = plt.subplots()
    diff_sem = [best_diff_semantic[w][1] if best_diff_semantic[w][1] is not None else 0 for w in WINDOWS]
    w2v_sem = [w2v_by_window[w]["semantic_acc"] if w in w2v_by_window else 0 for w in WINDOWS]
    ax.bar([i - width / 2 for i in x], diff_sem, width, label="diffusion_best")
    ax.bar([i + width / 2 for i in x], w2v_sem, width, label="word2vec")
    ax.set_xticks(x)
    ax.set_xticklabels([f"W={w}" for w in WINDOWS])
    ax.set_ylabel("Semantic accuracy (%)")
    ax.set_title("Best semantic accuracy by window: Word2Vec vs Diffusion")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(PLOTS_DIR, "best_semantic_by_window_word2vec_vs_diffusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Summary table (percentages to one decimal)
    print("\nSummary (per window):")
    print("-" * 80)
    print(f"{'Window':<8} {'Diffusion best iter':<20} {'Diffusion total %':<18} {'Word2Vec total %':<18} {'Diffusion semantic %':<22} {'Word2Vec semantic %':<20}")
    print("-" * 80)
    for w in WINDOWS:
        d_iter, d_total = best_diff_total.get(w, (None, None))
        d_sem_iter, d_sem = best_diff_semantic.get(w, (None, None))
        w2v_row = w2v_by_window.get(w, {})
        w2v_t = w2v_row.get("total_acc", None)
        w2v_s = w2v_row.get("semantic_acc", None)
        d_total_s = f"{d_total:.1f}" if d_total is not None else ""
        d_sem_s = f"{d_sem:.1f}" if d_sem is not None else ""
        w2v_t_s = f"{w2v_t:.1f}" if w2v_t is not None else ""
        w2v_s_s = f"{w2v_s:.1f}" if w2v_s is not None else ""
        print(f"{w:<8} {str(d_iter):<20} {d_total_s:<18} {w2v_t_s:<18} {d_sem_s:<22} {w2v_s_s:<20}")
    print("-" * 80)
    print(f"Plots saved under {PLOTS_DIR}")


if __name__ == "__main__":
    main()
