#!/usr/bin/env python3
"""
Evaluate text8 experiment outputs, build summary tables, write results/text8_results.csv,
and generate text8 plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
EVAL_ANALOGY = ROOT / "evaluation" / "evaluate_google_analogies.py"
EVAL_WORDSIM = ROOT / "evaluation" / "evaluate_wordsim.py"
QUESTIONS = ROOT / "evaluation" / "questions-words.txt"
WORDSIM = ROOT / "evaluation" / "wordsim353.txt"

UNDAMPED_BASE = ROOT / "data" / "iterative_vectors_text8"
ALPHA_BASE_PREFIX = ROOT / "data"
WORD2VEC_DIR = ROOT / "data" / "word2vec"
NNLM_PATH = ROOT / "data" / "nnlm" / "nnlm_text8_200d_ctx4.json"
RNNLM_PATH = ROOT / "data" / "rnnlm" / "rnnlm_text8_200d.json"

RESULTS_CSV = ROOT / "results" / "text8_results.csv"
PLOTS_DIR = ROOT / "results" / "plots"

ALPHAS = ["0.1", "0.3", "0.5", "0.7", "1.0"]
WINDOWS = [2, 4, 6, 8]
UNDAMPED_NS = [4, 8, 10]

ACC_SEM_RE = re.compile(r"Semantic\s+(\d+)/(\d+)\s+([\d.]+)%")
ACC_SYN_RE = re.compile(r"Syntactic\s+(\d+)/(\d+)\s+([\d.]+)%")
ACC_TOT_RE = re.compile(r"Total\s+(\d+)/(\d+)\s+([\d.]+)%")
WS_RE = re.compile(r"Spearman rho:\s+([\-\d.]+)\s+\(p=([\deE.+\-]+)\)")

CSV_FIELDS = [
    "method",
    "corpus",
    "dim",
    "alpha",
    "N_or_window",
    "iter",
    "semantic_acc",
    "syntactic_acc",
    "total_acc",
    "wordsim_rho",
    "semantic_valid",
    "syntactic_valid",
    "total_valid",
    "test_set",
]


@dataclass
class EvalRow:
    method: str
    corpus: str
    dim: int
    alpha: str
    n_or_window: str
    iteration: int
    semantic_acc: float
    syntactic_acc: float
    total_acc: float
    semantic_valid: int
    syntactic_valid: int
    total_valid: int
    test_set: str
    embedding_path: Path
    wordsim_rho: Optional[float] = None

    def to_csv(self) -> Dict[str, str]:
        return {
            "method": self.method,
            "corpus": self.corpus,
            "dim": str(self.dim),
            "alpha": self.alpha,
            "N_or_window": self.n_or_window,
            "iter": str(self.iteration),
            "semantic_acc": f"{self.semantic_acc:.1f}",
            "syntactic_acc": f"{self.syntactic_acc:.1f}",
            "total_acc": f"{self.total_acc:.1f}",
            "wordsim_rho": "" if self.wordsim_rho is None else f"{self.wordsim_rho:.3f}",
            "semantic_valid": str(self.semantic_valid),
            "syntactic_valid": str(self.syntactic_valid),
            "total_valid": str(self.total_valid),
            "test_set": self.test_set,
        }


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def parse_analogy_output(stdout: str) -> Optional[Tuple[int, int, int, float, float, float]]:
    sem_m = syn_m = tot_m = None
    for line in stdout.splitlines():
        if sem_m is None:
            sem_m = ACC_SEM_RE.search(line)
        if syn_m is None:
            syn_m = ACC_SYN_RE.search(line)
        if tot_m is None:
            tot_m = ACC_TOT_RE.search(line)

    if not (sem_m and syn_m and tot_m):
        return None

    sem_valid = int(sem_m.group(1))
    syn_valid = int(syn_m.group(1))
    tot_valid = int(tot_m.group(1))
    sem_acc = float(sem_m.group(3))
    syn_acc = float(syn_m.group(3))
    tot_acc = float(tot_m.group(3))
    return sem_valid, syn_valid, tot_valid, sem_acc, syn_acc, tot_acc


def run_analogy_eval(embedding_path: Path) -> Optional[Tuple[int, int, int, float, float, float, str]]:
    if not embedding_path.is_file():
        return None
    code, out, err = run_cmd([
        sys.executable,
        str(EVAL_ANALOGY),
        "--embeddings",
        str(embedding_path),
        "--questions",
        str(QUESTIONS),
    ])
    if code != 0:
        return None
    parsed = parse_analogy_output(out)
    if parsed is None:
        return None
    return (*parsed, out)


def run_wordsim_eval(embedding_path: Path) -> Optional[Tuple[float, str]]:
    if not embedding_path.is_file() or not WORDSIM.is_file():
        return None
    code, out, err = run_cmd([
        sys.executable,
        str(EVAL_WORDSIM),
        "--embeddings",
        str(embedding_path),
        "--wordsim",
        str(WORDSIM),
    ])
    if code != 0:
        return None
    m = WS_RE.search(out)
    if m is None:
        return None
    return float(m.group(1)), out


def evaluate_undamped(print_output: bool) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for n in UNDAMPED_NS:
        for i in range(0, 11):
            emb = UNDAMPED_BASE / f"v3_full_window_{n}" / f"{i}.json"
            res = run_analogy_eval(emb)
            if res is None:
                continue
            sem_valid, syn_valid, tot_valid, sem_acc, syn_acc, tot_acc, out = res
            if print_output:
                print(f"### N={n} iter {i}")
                print(out.strip())
            rows.append(EvalRow(
                method="diffusion",
                corpus="text8",
                dim=32,
                alpha="",
                n_or_window=str(n),
                iteration=i,
                semantic_acc=sem_acc,
                syntactic_acc=syn_acc,
                total_acc=tot_acc,
                semantic_valid=sem_valid,
                syntactic_valid=syn_valid,
                total_valid=tot_valid,
                test_set="questions-words.txt",
                embedding_path=emb,
            ))
    return rows


def evaluate_damped(print_output: bool) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for alpha in ALPHAS:
        for i in range(0, 21):
            emb = ALPHA_BASE_PREFIX / f"iterative_vectors_text8_alpha{alpha}" / "v3_full_window_8" / f"{i}.json"
            res = run_analogy_eval(emb)
            if res is None:
                continue
            sem_valid, syn_valid, tot_valid, sem_acc, syn_acc, tot_acc, out = res
            if print_output:
                print(f"### alpha={alpha} iter {i}")
                print(out.strip())
            rows.append(EvalRow(
                method="damped_diffusion",
                corpus="text8",
                dim=32,
                alpha=alpha,
                n_or_window="8",
                iteration=i,
                semantic_acc=sem_acc,
                syntactic_acc=syn_acc,
                total_acc=tot_acc,
                semantic_valid=sem_valid,
                syntactic_valid=syn_valid,
                total_valid=tot_valid,
                test_set="questions-words.txt",
                embedding_path=emb,
            ))
    return rows


def evaluate_word2vec(print_output: bool) -> List[EvalRow]:
    rows: List[EvalRow] = []
    patterns = [
        ("cbow", "word2vec_cbow"),
        ("sg", "word2vec_skipgram"),
    ]
    for mode, method_name in patterns:
        for w in WINDOWS:
            emb = WORD2VEC_DIR / f"word2vec_text8_{mode}_200d_window{w}.json"
            res = run_analogy_eval(emb)
            if res is None:
                continue
            sem_valid, syn_valid, tot_valid, sem_acc, syn_acc, tot_acc, out = res
            if print_output:
                print(f"### word2vec {mode} window {w}")
                print(out.strip())
            rows.append(EvalRow(
                method=method_name,
                corpus="text8",
                dim=200,
                alpha="",
                n_or_window=str(w),
                iteration=-1,
                semantic_acc=sem_acc,
                syntactic_acc=syn_acc,
                total_acc=tot_acc,
                semantic_valid=sem_valid,
                syntactic_valid=syn_valid,
                total_valid=tot_valid,
                test_set="questions-words.txt",
                embedding_path=emb,
            ))
    return rows


def evaluate_neural_lm(print_output: bool) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for method_name, emb in [("nnlm", NNLM_PATH), ("rnnlm", RNNLM_PATH)]:
        res = run_analogy_eval(emb)
        if res is None:
            continue
        sem_valid, syn_valid, tot_valid, sem_acc, syn_acc, tot_acc, out = res
        if print_output:
            print(f"### {method_name.upper()} text8")
            print(out.strip())
        rows.append(EvalRow(
            method=method_name,
            corpus="text8",
            dim=200,
            alpha="",
            n_or_window="",
            iteration=-1,
            semantic_acc=sem_acc,
            syntactic_acc=syn_acc,
            total_acc=tot_acc,
            semantic_valid=sem_valid,
            syntactic_valid=syn_valid,
            total_valid=tot_valid,
            test_set="questions-words.txt",
            embedding_path=emb,
        ))
    return rows


def pick_best(rows: List[EvalRow]) -> Optional[EvalRow]:
    if not rows:
        return None
    return max(rows, key=lambda r: (r.total_acc, r.semantic_acc, -r.iteration))


def print_method_table(final_rows: List[Tuple[str, Optional[EvalRow]]]) -> None:
    print("Method              | Dim  | Semantic % | Syntactic % | Total % | WordSim rho")
    print("--------------------|------|------------|-------------|---------|------------")
    for label, row in final_rows:
        if row is None:
            print(f"{label:<20}| {'-':>4} | {'-':>10} | {'-':>11} | {'-':>7} | {'-':>10}")
            continue
        ws = "-" if row.wordsim_rho is None else f"{row.wordsim_rho:.3f}"
        print(
            f"{label:<20}| {row.dim:>4} | {row.semantic_acc:>10.1f} | {row.syntactic_acc:>11.1f} | {row.total_acc:>7.1f} | {ws:>10}"
        )


def print_alpha_table(alpha_best: Dict[str, Optional[EvalRow]]) -> None:
    print("Alpha | Best Iter | Semantic % | Syntactic % | Total %")
    print("------|-----------|------------|-------------|--------")
    for alpha in ALPHAS:
        row = alpha_best.get(alpha)
        if row is None:
            print(f"{alpha:<5} | {'-':>9} | {'-':>10} | {'-':>11} | {'-':>6}")
            continue
        print(
            f"{alpha:<5} | {row.iteration:>9} | {row.semantic_acc:>10.1f} | {row.syntactic_acc:>11.1f} | {row.total_acc:>6.1f}"
        )


def print_undamped_table(undamped_rows: List[EvalRow]) -> None:
    print("N | Best Iter | Semantic % | Syntactic % | Total %")
    print("--|-----------|------------|-------------|--------")
    for n in UNDAMPED_NS:
        sub = [r for r in undamped_rows if r.n_or_window == str(n)]
        if not sub:
            print(f"{n:<1} | {'-':>9} | {'-':>10} | {'-':>11} | {'-':>6}")
            continue
        best = pick_best(sub)
        assert best is not None
        print(
            f"{n:<1} | {best.iteration:>9} | {best.semantic_acc:>10.1f} | {best.syntactic_acc:>11.1f} | {best.total_acc:>6.1f}"
        )


def save_csv(rows: List[EvalRow]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_csv())


def make_plot_damped(alpha_rows: List[EvalRow]) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    for alpha in ALPHAS:
        sub = sorted([r for r in alpha_rows if r.alpha == alpha], key=lambda r: r.iteration)
        if not sub:
            continue
        ax.plot([r.iteration for r in sub], [r.semantic_acc for r in sub], marker="o", label=f"alpha={alpha}")

    ax.set_title("Effect of Damping on Semantic Accuracy (text8, N=8)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Semantic Accuracy (%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = PLOTS_DIR / "text8_damped_diffusion.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def make_plot_method_comparison(final_rows: List[Tuple[str, Optional[EvalRow]]]) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [name for name, row in final_rows if row is not None]
    vals = [row.total_acc for name, row in final_rows if row is not None and row is not None]

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    bars = ax.bar(labels, vals)
    ax.set_title("Text8 Method Comparison (Best Config per Method)")
    ax.set_ylabel("Total Accuracy (%)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    path = PLOTS_DIR / "text8_method_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def make_plot_undamped_curve(undamped_rows: List[EvalRow]) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    for n in UNDAMPED_NS:
        sub = sorted([r for r in undamped_rows if r.n_or_window == str(n)], key=lambda r: r.iteration)
        if not sub:
            continue
        x = [r.iteration for r in sub]
        axes[0].plot(x, [r.semantic_acc for r in sub], marker="o", label=f"N={n}")
        axes[1].plot(x, [r.syntactic_acc for r in sub], marker="o", label=f"N={n}")

    axes[0].set_title("Semantic Accuracy vs Iteration")
    axes[1].set_title("Syntactic Accuracy vs Iteration")
    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(list(range(0, 11)))
        ax.legend()

    fig.suptitle("Diffusion Embedding Accuracy by Iteration (text8, undamped)")
    path = PLOTS_DIR / "text8_accuracy_curve.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build text8 evaluation tables, CSV, and plots")
    parser.add_argument("--print-output", action="store_true", help="Print full evaluation outputs")
    args = parser.parse_args()

    undamped_rows = evaluate_undamped(print_output=args.print_output)
    alpha_rows = evaluate_damped(print_output=args.print_output)
    w2v_rows = evaluate_word2vec(print_output=args.print_output)
    lm_rows = evaluate_neural_lm(print_output=args.print_output)

    cbow_rows = [r for r in w2v_rows if r.method == "word2vec_cbow"]
    sg_rows = [r for r in w2v_rows if r.method == "word2vec_skipgram"]

    best_diffusion = pick_best(undamped_rows)
    best_cbow = pick_best(cbow_rows)
    best_sg = pick_best(sg_rows)
    best_damped = pick_best(alpha_rows)
    best_nnlm = pick_best([r for r in lm_rows if r.method == "nnlm"])
    best_rnnlm = pick_best([r for r in lm_rows if r.method == "rnnlm"])

    alpha_best: Dict[str, Optional[EvalRow]] = {}
    for alpha in ALPHAS:
        alpha_best[alpha] = pick_best([r for r in alpha_rows if r.alpha == alpha])

    # User-specified WordSim scope: best diffusion, best CBOW, best skip-gram only.
    for row in [best_diffusion, best_cbow, best_sg]:
        if row is None:
            continue
        ws = run_wordsim_eval(row.embedding_path)
        if ws is not None:
            rho, out = ws
            row.wordsim_rho = rho
            if args.print_output:
                print(f"### WordSim {row.method} {row.embedding_path}")
                print(out.strip())

    all_rows = undamped_rows + alpha_rows + w2v_rows + lm_rows
    save_csv(all_rows)

    final_rows: List[Tuple[str, Optional[EvalRow]]] = [
        ("RNNLM", best_rnnlm),
        ("NNLM", best_nnlm),
        ("CBOW (best W)", best_cbow),
        ("Skip-gram (best W)", best_sg),
        ("Diffusion (best)", best_diffusion),
        ("Damped (best alpha)", best_damped),
    ]

    print()
    print_method_table(final_rows)
    print()
    print_alpha_table(alpha_best)
    print()
    print_undamped_table(undamped_rows)

    damped_plot = make_plot_damped(alpha_rows)
    method_plot = make_plot_method_comparison(final_rows)
    curve_plot = make_plot_undamped_curve(undamped_rows)

    print()
    print(f"Saved CSV: {RESULTS_CSV}")
    print(f"Saved plot: {damped_plot}")
    print(f"Saved plot: {method_plot}")
    print(f"Saved plot: {curve_plot}")


if __name__ == "__main__":
    main()
