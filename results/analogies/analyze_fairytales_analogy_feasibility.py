#!/usr/bin/env python3
"""
Feasibility analysis for building a Fairytales-compatible analogy benchmark
using the shared vocabulary intersection across the main Diffusion + Word2Vec
embeddings (no retraining, no scoring changes).

Writes:
  - results/analogies/fairytales_analogy_feasibility.md
  - results/analogies/candidate_semantic_pairs.json
  - results/analogies/candidate_syntactic_pairs.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_PATHS = {
    # Diffusion representative checkpoints (as requested)
    "diffusion_N2": ROOT / "data" / "iterative_vectors" / "window_2_iter_0_v3_32bit.json",
    "diffusion_N4": ROOT / "data" / "iterative_vectors" / "window_4_iter_1_v3_32bit.json",
    "diffusion_N6": ROOT / "data" / "iterative_vectors" / "window_6_iter_0_v3_32bit.json",
    "diffusion_N8": ROOT / "data" / "iterative_vectors" / "window_8_iter_0_v3_32bit.json",
    # Word2Vec baselines
    "word2vec_w2": ROOT / "data" / "word2vec" / "word2vec_vectors_32d_window2.json",
    "word2vec_w4": ROOT / "data" / "word2vec" / "word2vec_vectors_32d_window4.json",
    "word2vec_w6": ROOT / "data" / "word2vec" / "word2vec_vectors_32d_window6.json",
    "word2vec_w8": ROOT / "data" / "word2vec" / "word2vec_vectors_32d_window8.json",
}

OUT_MD = ROOT / "results" / "analogies" / "fairytales_analogy_feasibility.md"
OUT_SEM = ROOT / "results" / "analogies" / "candidate_semantic_pairs.json"
OUT_SYN = ROOT / "results" / "analogies" / "candidate_syntactic_pairs.json"


def load_vocab(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.keys())


def relation_pairs_by_suffix(vocab: set[str], suffix: str) -> list[tuple[str, str]]:
    pairs = []
    for w in vocab:
        if w.endswith(suffix):
            base = w[: -len(suffix)]
            if base and base in vocab:
                pairs.append((base, w))
    return pairs


def relation_pairs_plural_s(vocab: set[str]) -> list[tuple[str, str]]:
    pairs = []
    for w in vocab:
        if w.endswith("s") and len(w) > 3:
            base = w[:-1]
            if base in vocab:
                pairs.append((base, w))
    return pairs


def relation_pairs_plural_es(vocab: set[str]) -> list[tuple[str, str]]:
    pairs = []
    for w in vocab:
        if w.endswith("es") and len(w) > 4:
            base = w[:-2]
            if base in vocab:
                pairs.append((base, w))
    return pairs


def relation_pairs_y_ies(vocab: set[str]) -> list[tuple[str, str]]:
    pairs = []
    for w in vocab:
        if w.endswith("ies") and len(w) > 4:
            base = w[:-3] + "y"
            if base in vocab:
                pairs.append((base, w))
    return pairs


def canonicalize_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    # Deduplicate while preserving deterministic order
    seen = set()
    out = []
    for a, b in sorted(pairs):
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out


def main():
    # Load vocab intersection
    vocabs = {}
    for name, p in EMBEDDING_PATHS.items():
        if not p.is_file():
            raise SystemExit(f"Missing embedding file required for analysis: {p}")
        vocabs[name] = load_vocab(p)
    shared = None
    for v in vocabs.values():
        shared = v if shared is None else (shared & v)
    assert shared is not None

    # Candidate semantic pairs: simple curated relations if both words exist.
    semantic_pair_candidates = [
        ("king", "queen"),
        ("prince", "princess"),
        ("man", "woman"),
        ("boy", "girl"),
        ("father", "mother"),
        ("brother", "sister"),
        ("son", "daughter"),
        ("husband", "wife"),
        ("male", "female"),
        ("old", "young"),
        ("black", "white"),
        ("day", "night"),
        ("sun", "moon"),
        ("good", "bad"),
        ("big", "small"),
    ]
    semantic_pairs = [(a, b) for (a, b) in semantic_pair_candidates if a in shared and b in shared]

    # Candidate syntactic/morphological pairs from shared vocab.
    syntactic_pairs_by_relation: dict[str, list[tuple[str, str]]] = {}
    syntactic_pairs_by_relation["plural_s"] = canonicalize_pairs(relation_pairs_plural_s(shared))
    syntactic_pairs_by_relation["plural_es"] = canonicalize_pairs(relation_pairs_plural_es(shared))
    syntactic_pairs_by_relation["plural_y_ies"] = canonicalize_pairs(relation_pairs_y_ies(shared))
    syntactic_pairs_by_relation["suffix_ed"] = canonicalize_pairs(relation_pairs_by_suffix(shared, "ed"))
    syntactic_pairs_by_relation["suffix_ing"] = canonicalize_pairs(relation_pairs_by_suffix(shared, "ing"))
    syntactic_pairs_by_relation["suffix_er"] = canonicalize_pairs(relation_pairs_by_suffix(shared, "er"))
    syntactic_pairs_by_relation["suffix_est"] = canonicalize_pairs(relation_pairs_by_suffix(shared, "est"))

    # Some small hand-curated irregular morph pairs if present in shared vocab.
    irregular = [
        ("go", "went"),
        ("take", "took"),
        ("give", "gave"),
        ("come", "came"),
        ("run", "ran"),
        ("see", "saw"),
        ("say", "said"),
        ("get", "got"),
        ("make", "made"),
        ("find", "found"),
        ("think", "thought"),
        ("tell", "told"),
    ]
    syntactic_pairs_by_relation["irregular_past"] = [(a, b) for (a, b) in irregular if a in shared and b in shared]

    # Summarize feasibility for analogy construction:
    # For a relation to support k analogies, we need at least k+1 pairs ideally; but as a quick heuristic,
    # count how many distinct pairs exist per relation.
    relation_counts = {k: len(v) for k, v in syntactic_pairs_by_relation.items()}

    # Write machine-readable JSON helpers
    OUT_SEM.parent.mkdir(parents=True, exist_ok=True)
    OUT_SEM.write_text(json.dumps({"shared_vocab_size": len(shared), "pairs": semantic_pairs}, indent=2), encoding="utf-8")
    OUT_SYN.write_text(
        json.dumps({"shared_vocab_size": len(shared), "relations": syntactic_pairs_by_relation}, indent=2),
        encoding="utf-8",
    )

    # Create markdown report
    lines: list[str] = []
    lines.append("# Fairytales analogy feasibility report (shared vocab)")
    lines.append("")
    lines.append("## Shared vocabulary intersection")
    lines.append(f"- Shared vocab size (intersection over {len(EMBEDDING_PATHS)} embeddings): **{len(shared)}**")
    lines.append("")
    lines.append("## Candidate semantic relation pairs (curated)")
    lines.append(f"- Found **{len(semantic_pairs)}** curated semantic pairs present in shared vocab.")
    if semantic_pairs:
        lines.append("")
        for a, b in semantic_pairs:
            lines.append(f"- `{a}` ↔ `{b}`")
    lines.append("")
    lines.append("## Candidate syntactic relation pairs (heuristic mining)")
    lines.append("Counts below are the number of word pairs `(base, variant)` found in shared vocab for each relation.")
    lines.append("")
    lines.append("| Relation | Pair count | Examples |")
    lines.append("|----------|------------|----------|")
    for rel in sorted(syntactic_pairs_by_relation.keys()):
        pairs = syntactic_pairs_by_relation[rel]
        ex = ", ".join([f"{a}->{b}" for (a, b) in pairs[:5]])
        lines.append(f"| `{rel}` | {len(pairs)} | {ex} |")
    lines.append("")
    lines.append("## Feasibility decision")
    lines.append("- Minimum target to build a usable benchmark: **≥10 semantic analogies and ≥10 syntactic analogies**.")
    lines.append("- This script only measures whether the *raw ingredients* exist (pairs), not whether they form clean non-duplicate analogy quadruples.")
    lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_SEM}")
    print(f"Wrote {OUT_SYN}")
    print(f"Syntactic relation pair counts: {relation_counts}")


if __name__ == "__main__":
    main()

