from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

try:
    from adjustText import adjust_text
except Exception:  # pragma: no cover - optional dependency
    adjust_text = None


def choose_diffusion_embedding() -> Tuple[Path, List[str]]:
    base = Path("data/iterative_vectors")
    primary = base / "window_10_iter_1_v3_32bit.json"
    fallback = base / "window_8_iter_1_v3_32bit.json"

    if primary.exists():
        return primary, []
    if fallback.exists():
        return fallback, [f"Missing preferred diffusion embedding: {primary.as_posix()}"]

    candidates = sorted(base.glob("window_*_iter_1_v3_32bit.json"))
    if candidates:
        notes = [f"Missing preferred diffusion embedding: {primary.as_posix()}"]
        notes.append(f"Missing fallback diffusion embedding: {fallback.as_posix()}")
        notes.append("Available iterative_vectors candidates:")
        notes.extend(f"- {p.as_posix()}" for p in candidates)
        notes.append(f"Using closest available match: {candidates[0].as_posix()}")
        return candidates[0], notes

    existing = sorted(p.as_posix() for p in base.glob("*.json"))
    notes = [
        f"Missing preferred diffusion embedding: {primary.as_posix()}",
        f"Missing fallback diffusion embedding: {fallback.as_posix()}",
        "No window_*_iter_1_v3_32bit.json candidates found.",
        "Existing JSON files in data/iterative_vectors:",
    ]
    notes.extend(f"- {name}" for name in existing)
    raise FileNotFoundError("\n".join(notes))


def choose_cbow_embedding() -> Tuple[Path, List[str]]:
    base = Path("data/word2vec")
    primary = base / "word2vec_cbow_200d_window4.json"

    if primary.exists():
        return primary, []

    candidates = sorted(base.glob("word2vec_cbow_200d_window*.json"))
    if candidates:
        def parse_window(path: Path) -> int:
            name = path.stem
            marker = "window"
            if marker not in name:
                return 9999
            tail = name.split(marker, 1)[1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            return int(digits) if digits else 9999

        closest = sorted(candidates, key=lambda p: abs(parse_window(p) - 4))[0]
        notes = [f"Missing preferred CBOW embedding: {primary.as_posix()}"]
        notes.append("Available CBOW candidates:")
        notes.extend(f"- {p.as_posix()}" for p in candidates)
        notes.append(f"Using closest available match: {closest.as_posix()}")
        return closest, notes

    existing = sorted(p.as_posix() for p in base.glob("*.json"))
    notes = [
        f"Missing preferred CBOW embedding: {primary.as_posix()}",
        "No word2vec_cbow_200d_window*.json candidates found.",
        "Existing JSON files in data/word2vec:",
    ]
    notes.extend(f"- {name}" for name in existing)
    raise FileNotFoundError("\n".join(notes))


def load_embedding(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}


def build_matrix(embedding: Dict[str, np.ndarray], words: Sequence[str]) -> np.ndarray:
    mat = np.asarray([embedding[w] for w in words], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    return mat


def group_words_in_intersection(word_groups: Dict[str, List[str]], intersection: set[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for group, words in word_groups.items():
        keep = [w for w in words if w in intersection]
        grouped[group] = keep
    return grouped


def plot_embedding(
    ax: plt.Axes,
    coords: np.ndarray,
    words: Sequence[str],
    grouped_words: Dict[str, List[str]],
    colors: Dict[str, Tuple[float, float, float, float]],
    title: str,
    show_legend: bool,
) -> List[Line2D]:
    idx = {w: i for i, w in enumerate(words)}
    handles: List[Line2D] = []

    ax.scatter(coords[:, 0], coords[:, 1], s=8, c="#b3b3b3", alpha=0.35, edgecolors="none")

    text_artists = []
    for group, group_list in grouped_words.items():
        if not group_list:
            continue

        c = colors[group]
        xy = np.asarray([coords[idx[w]] for w in group_list], dtype=np.float32)
        ax.scatter(xy[:, 0], xy[:, 1], s=38, c=[c], alpha=0.95, edgecolors="black", linewidths=0.3)

        for w in group_list:
            x, y = coords[idx[w]]
            text_artists.append(
                ax.text(x + 0.01, y + 0.01, w, fontsize=8, color=c)
            )

        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=c,
                markeredgecolor="black",
                markeredgewidth=0.3,
                markersize=7,
                label=group,
            )
        )

    if adjust_text is not None and text_artists:
        adjust_text(
            text_artists,
            ax=ax,
            only_move={"points": "xy", "text": "xy"},
            arrowprops=dict(arrowstyle="-", lw=0.4, color="#444444"),
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("PC1", fontsize=12)
    ax.set_ylabel("PC2", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)

    if show_legend and handles:
        ax.legend(handles=handles, fontsize=10, loc="best")

    return handles


def mean_pairwise_cosine(words: Sequence[str], matrix: np.ndarray, index: Dict[str, int]) -> float:
    if len(words) < 2:
        return float("nan")

    sub = matrix[np.asarray([index[w] for w in words], dtype=np.int32)]
    sims = sub @ sub.T
    mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
    vals = sims[mask]
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def main() -> None:
    np.random.seed(42)

    word_groups = {
        "Royalty": ["king", "queen", "prince", "princess", "kingdom", "throne", "crown", "royal"],
        "Family": ["mother", "father", "son", "daughter", "brother", "sister", "child", "children"],
        "Animals": ["wolf", "bear", "fox", "horse", "bird", "cat", "dog", "lion"],
        "Magic": ["magic", "witch", "fairy", "spell", "enchanted", "wizard", "curse"],
        "Nature": ["forest", "tree", "river", "mountain", "garden", "sea", "water", "stone"],
    }

    diffusion_path, diffusion_notes = choose_diffusion_embedding()
    cbow_path, cbow_notes = choose_cbow_embedding()

    diff_embed = load_embedding(diffusion_path)
    cbow_embed = load_embedding(cbow_path)

    inter = sorted(set(diff_embed.keys()) & set(cbow_embed.keys()))
    if len(inter) < 2:
        raise RuntimeError("Intersection vocabulary too small for PCA.")

    grouped_words = group_words_in_intersection(word_groups, set(inter))

    diff_mat = build_matrix(diff_embed, inter)
    cbow_mat = build_matrix(cbow_embed, inter)

    pca_diff = PCA(n_components=2, svd_solver="randomized", random_state=42)
    pca_cbow = PCA(n_components=2, svd_solver="randomized", random_state=42)
    diff_2d = pca_diff.fit_transform(diff_mat)
    cbow_2d = pca_cbow.fit_transform(cbow_mat)

    cmap = plt.get_cmap("tab10")
    colors = {group: cmap(i) for i, group in enumerate(word_groups.keys())}

    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Side-by-side comparison plot.
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    handles = plot_embedding(
        axes[0],
        diff_2d,
        inter,
        grouped_words,
        colors,
        "Diffusion (N=10, iter 1)",
        show_legend=False,
    )
    plot_embedding(
        axes[1],
        cbow_2d,
        inter,
        grouped_words,
        colors,
        "CBOW (W=4)",
        show_legend=False,
    )

    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(5, len(handles)),
            fontsize=10,
            bbox_to_anchor=(0.5, -0.02),
        )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path_comparison = out_dir / "pca_comparison.png"
    fig.savefig(path_comparison, dpi=300)
    plt.close(fig)

    # Individual Diffusion plot.
    fig_d, ax_d = plt.subplots(figsize=(8, 7), dpi=300)
    plot_embedding(
        ax_d,
        diff_2d,
        inter,
        grouped_words,
        colors,
        "Diffusion (N=10, iter 1)",
        show_legend=True,
    )
    fig_d.tight_layout()
    path_diff = out_dir / "pca_diffusion.png"
    fig_d.savefig(path_diff, dpi=300)
    plt.close(fig_d)

    # Individual CBOW plot.
    fig_c, ax_c = plt.subplots(figsize=(8, 7), dpi=300)
    plot_embedding(
        ax_c,
        cbow_2d,
        inter,
        grouped_words,
        colors,
        "CBOW (W=4)",
        show_legend=True,
    )
    fig_c.tight_layout()
    path_cbow = out_dir / "pca_cbow.png"
    fig_c.savefig(path_cbow, dpi=300)
    plt.close(fig_c)

    idx = {w: i for i, w in enumerate(inter)}

    lines: List[str] = []
    lines.append("=== PCA COMPARISON SETUP ===")
    lines.append(f"Diffusion embedding: {diffusion_path.as_posix()}")
    lines.append(f"CBOW embedding:      {cbow_path.as_posix()}")
    lines.extend(diffusion_notes + cbow_notes)
    lines.append(f"Intersection vocabulary size: {len(inter)}")
    lines.append(f"adjustText available: {'yes' if adjust_text is not None else 'no'}")
    lines.append("")

    lines.append("=== GROUP WORDS USED (present in BOTH embeddings) ===")
    for group, words in grouped_words.items():
        joined = ", ".join(words) if words else "(none)"
        lines.append(f"{group}: {joined}")
    lines.append("")

    lines.append("=== CLUSTER COHERENCE (mean pairwise cosine within group) ===")
    lines.append(f"{'Group':<12} {'Diffusion':>10} {'CBOW':>10}")
    for group, words in grouped_words.items():
        d_val = mean_pairwise_cosine(words, diff_mat, idx)
        c_val = mean_pairwise_cosine(words, cbow_mat, idx)
        d_txt = f"{d_val:.3f}" if np.isfinite(d_val) else "nan"
        c_txt = f"{c_val:.3f}" if np.isfinite(c_val) else "nan"
        lines.append(f"{group:<12} {d_txt:>10} {c_txt:>10}")

    lines.append("")
    lines.append("Saved PNG files:")
    lines.append(f"- {path_comparison.as_posix()}")
    lines.append(f"- {path_diff.as_posix()}")
    lines.append(f"- {path_cbow.as_posix()}")

    text = "\n".join(lines)
    print(text)

    out_txt = Path("results/pca_comparison_report.txt")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved report: {out_txt.as_posix()}")


if __name__ == "__main__":
    main()
