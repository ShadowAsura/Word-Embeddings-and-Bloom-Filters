"""
Load numeric JSONs from a directory (e.g. v3_full_window_4/0.json..399.json),
compute mean L2 between consecutive iterations, write CSV + PNG.
Usage: python evaluation/compute_stability_from_dir.py --dir data/iterative_vectors/v3_full_window_4
"""
import argparse
import json
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description="Stability from a dir of numeric iteration JSONs")
    p.add_argument("--dir", required=True, help="Directory containing 0.json, 1.json, ...")
    args = p.parse_args()
    data_dir = os.path.abspath(args.dir)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Not a directory: {data_dir}")

    # Discover numeric JSONs and sort by iteration
    files = [f for f in os.listdir(data_dir) if f.endswith(".json") and f.replace(".json", "").isdigit()]
    iters = sorted([int(f.replace(".json", "")) for f in files])
    if len(iters) < 2:
        raise SystemExit(f"Need at least 2 numeric JSONs in {data_dir}")
    print(f"Found {len(iters)} files: {iters[0]}.json .. {iters[-1]}.json")

    def load_embedding(t):
        path = os.path.join(data_dir, f"{t}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        words = list(data.keys())
        V = np.array([data[w] for w in words], dtype=np.float64)
        return words, V

    # Load all
    loaded = []
    for t in tqdm(iters, desc="Loading JSONs", unit="file"):
        words, V = load_embedding(t)
        loaded.append((t, words, V))
    print(f"Loaded {len(loaded)} embeddings.")

    # Align to first file's word list
    ref_words = loaded[0][1]
    aligned = []
    for t, words, V in tqdm(loaded, desc="Aligning vocab", unit="file"):
        word_to_col = {w: j for j, w in enumerate(words)}
        try:
            idx = [word_to_col[w] for w in ref_words]
        except KeyError:
            continue
        aligned.append((t, V[idx]))

    # Consecutive pairs: mean L2 (Δ=1)
    rows = []
    n_pairs = len(aligned) - 1
    for i in tqdm(range(n_pairs), desc="Computing pairs", unit="pair"):
        t_a, V_a = aligned[i]
        t_b, V_b = aligned[i + 1]
        delta = t_b - t_a
        if delta <= 0:
            continue
        diff = V_b - V_a
        mean_l2 = float(np.mean(np.linalg.norm(diff, axis=1)))
        mean_l2_per_iter = mean_l2 / delta
        rows.append({"iterA": t_a, "iterB": t_b, "delta": delta, "meanL2": mean_l2, "meanL2_per_iter": mean_l2_per_iter})
    print(f"Computed {len(rows)} consecutive pairs.")

    # Output name from dir basename (e.g. v3_full_window_4)
    base_name = os.path.basename(data_dir.rstrip(os.sep))
    out_csv = os.path.join(SCRIPT_DIR, f"{base_name}_stability.csv")
    out_png = os.path.join(SCRIPT_DIR, f"{base_name}_stability.png")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iterA", "iterB", "delta", "meanL2", "meanL2_per_iter"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")

    fig, ax = plt.subplots()
    x = [r["iterB"] for r in rows]
    y = [r["meanL2_per_iter"] for r in rows]
    ax.plot(x, y, "b.-", markersize=3)
    ax.set_xlabel("iterB")
    ax.set_ylabel("meanL2_per_iter")
    ax.set_title(f"Stability (Δ=1): {base_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
