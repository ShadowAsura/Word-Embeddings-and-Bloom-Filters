#!/usr/bin/env python3
"""Quick 2D PCA of an embedding JSON. Saves plot to evaluation/pca_embedding.png."""
import argparse
import json
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True, help="Path to embedding JSON")
    parser.add_argument("--out", default=None, help="Output PNG path")
    parser.add_argument("--max-points", type=int, default=2000, help="Subsample for plot (default 2000)")
    args = parser.parse_args()
    with open(args.embeddings, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = list(data.keys())
    V = np.array([data[w] for w in words], dtype=np.float64)
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1
    V = V / norms
    from sklearn.decomposition import PCA
    n = V.shape[0]
    if n > args.max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, args.max_points, replace=False)
        V_plot = V[idx]
        words_plot = [words[i] for i in idx]
    else:
        V_plot = V
        words_plot = words
    pca = PCA(n_components=2)
    X = pca.fit_transform(V_plot)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot. Install with: pip install matplotlib")
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, s=8)
    for i, w in enumerate(words_plot[:50]):
        ax.annotate(w, (X[i, 0], X[i, 1]), fontsize=6, alpha=0.8)
    ax.set_title(f"PCA of embeddings: {os.path.basename(args.embeddings)}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    out = args.out or os.path.join(os.path.dirname(args.embeddings), "pca_embedding.png")
    out = os.path.normpath(out)
    if not out.endswith(".png"):
        out = out + ".png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

if __name__ == "__main__":
    main()
