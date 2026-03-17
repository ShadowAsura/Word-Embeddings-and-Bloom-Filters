#!/usr/bin/env python3
"""Copy diffusion checkpoints into per-window dirs as 0.json, 1.json, 4.json, ... then run stability."""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_DIR = os.path.join(ROOT, "data", "iterative_vectors")
STABILITY_DIR = os.path.join(ROOT, "results", "stability")
CHECKPOINTS = [0, 1, 4, 9, 24, 49, 74, 99, 149, 399]

def main():
    os.makedirs(STABILITY_DIR, exist_ok=True)
    for window in [2, 4, 6, 8]:
        out_dir = os.path.join(STABILITY_DIR, f"checkpoints_window_{window}")
        os.makedirs(out_dir, exist_ok=True)
        for it in CHECKPOINTS:
            src = os.path.join(VEC_DIR, f"window_{window}_iter_{it}_v3_32bit.json")
            if not os.path.isfile(src):
                continue
            dst = os.path.join(out_dir, f"{it}.json")
            shutil.copy2(src, dst)
        print(f"  {out_dir}: {len([f for f in os.listdir(out_dir) if f.endswith('.json')])} files")

if __name__ == "__main__":
    main()
