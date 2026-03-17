#!/usr/bin/env python3
"""
Build a shared-valid analogy question file where all 4 words in each analogy
exist in every embedding used in the main comparison table.

Reads:
  - evaluation/analogy_questions.json
  - diffusion checkpoints:
      data/iterative_vectors/window_2_iter_0_v3_32bit.json
      data/iterative_vectors/window_4_iter_1_v3_32bit.json
      data/iterative_vectors/window_6_iter_0_v3_32bit.json
      data/iterative_vectors/window_8_iter_0_v3_32bit.json
  - word2vec baselines:
      data/word2vec/word2vec_vectors_32d_window2.json
      data/word2vec/word2vec_vectors_32d_window4.json
      data/word2vec/word2vec_vectors_32d_window6.json
      data/word2vec/word2vec_vectors_32d_window8.json

Writes:
  - evaluation/analogy_questions_shared.json
  - results/analogies/shared_analogy_filter_summary.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALOGY_PATH = ROOT / "evaluation" / "analogy_questions.json"
SHARED_PATH = ROOT / "evaluation" / "analogy_questions_shared.json"
SUMMARY_MD = ROOT / "results" / "analogies" / "shared_analogy_filter_summary.md"

EMBEDDING_PATHS = [
    ROOT / "data" / "iterative_vectors" / "window_2_iter_0_v3_32bit.json",
    ROOT / "data" / "iterative_vectors" / "window_4_iter_1_v3_32bit.json",
    ROOT / "data" / "iterative_vectors" / "window_6_iter_0_v3_32bit.json",
    ROOT / "data" / "iterative_vectors" / "window_8_iter_0_v3_32bit.json",
    ROOT / "data" / "word2vec" / "word2vec_vectors_32d_window2.json",
    ROOT / "data" / "word2vec" / "word2vec_vectors_32d_window4.json",
    ROOT / "data" / "word2vec" / "word2vec_vectors_32d_window6.json",
    ROOT / "data" / "word2vec" / "word2vec_vectors_32d_window8.json",
]


def load_vocab(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.keys())


def main() -> None:
    with ANALOGY_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    sem_orig = questions.get("semantic", [])
    syn_orig = questions.get("syntactic", [])

    # Load vocab sets for all embeddings
    vocabs = []
    for p in EMBEDDING_PATHS:
        if not p.is_file():
            raise SystemExit(f"Missing embedding file required for shared-vocab filtering: {p}")
        vocabs.append(load_vocab(p))

    def words_present_in_all(words: list[str]) -> tuple[bool, list[str]]:
        missing: list[str] = []
        for w in words:
            for vocab in vocabs:
                if w not in vocab:
                    missing.append(w)
                    break
        return (len(missing) == 0, missing)

    sem_kept = []
    syn_kept = []
    removed = {"semantic": [], "syntactic": []}

    for q in sem_orig:
        if len(q) != 4:
            continue
        a, b, c, d = q
        ok, missing = words_present_in_all([a, b, c, d])
        if ok:
            sem_kept.append(q)
        else:
            removed["semantic"].append({"question": q, "missing_words": sorted(set(missing))})

    for q in syn_orig:
        if len(q) != 4:
            continue
        a, b, c, d = q
        ok, missing = words_present_in_all([a, b, c, d])
        if ok:
            syn_kept.append(q)
        else:
            removed["syntactic"].append({"question": q, "missing_words": sorted(set(missing))})

    shared = {"semantic": sem_kept, "syntactic": syn_kept}
    SHARED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SHARED_PATH.open("w", encoding="utf-8") as f:
        json.dump(shared, f, indent=2)

    # Write summary markdown
    RESULTS_DIR = SUMMARY_MD.parent
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Shared analogy filter summary")
    lines.append("")
    lines.append(f"- Original semantic questions: {len(sem_orig)}")
    lines.append(f"- Original syntactic questions: {len(syn_orig)}")
    lines.append(f"- Kept semantic questions: {len(sem_kept)}")
    lines.append(f"- Kept syntactic questions: {len(syn_kept)}")
    lines.append("")

    def section(cat: str) -> None:
        if not removed[cat]:
            return
        lines.append(f"## Removed {cat} questions")
        lines.append("")
        for entry in removed[cat]:
            q = entry["question"]
            missing = ", ".join(entry["missing_words"])
            lines.append(f"- `{q}` (missing in at least one embedding: {missing})")
        lines.append("")

    section("semantic")
    section("syntactic")

    with SUMMARY_MD.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {SHARED_PATH}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()

