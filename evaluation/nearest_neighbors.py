from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


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


def load_embedding(path: Path) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    words = list(data.keys())
    vectors = np.asarray([data[w] for w in words], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    vectors = vectors / norms
    index = {w: i for i, w in enumerate(words)}
    return words, vectors, index


def top_k_neighbors(
    query: str,
    words: Sequence[str],
    vectors: np.ndarray,
    index: Dict[str, int],
    k: int,
) -> List[Tuple[str, float]]:
    q_idx = index[query]
    sims = vectors @ vectors[q_idx]
    sims[q_idx] = -np.inf

    if k >= len(words):
        sorted_idx = np.argsort(-sims)
    else:
        top_idx = np.argpartition(-sims, k)[:k]
        sorted_idx = top_idx[np.argsort(-sims[top_idx])]

    return [(words[i], float(sims[i])) for i in sorted_idx[:k]]


def format_neighbor(neighbors: List[Tuple[str, float]], rank: int) -> str:
    if rank <= len(neighbors):
        word, score = neighbors[rank - 1]
        return f"{rank}. {word} ({score:.3f})"
    return f"{rank}. -"


def main() -> None:
    np.random.seed(42)

    queries = [
        "king",
        "queen",
        "prince",
        "princess",
        "wolf",
        "forest",
        "good",
        "evil",
        "beautiful",
        "old",
        "mother",
        "father",
        "child",
        "dragon",
        "magic",
        "sword",
        "brave",
        "dark",
        "little",
        "death",
    ]

    diffusion_path, diffusion_notes = choose_diffusion_embedding()
    cbow_path, cbow_notes = choose_cbow_embedding()

    diff_words, diff_vecs, diff_index = load_embedding(diffusion_path)
    cbow_words, cbow_vecs, cbow_index = load_embedding(cbow_path)

    lines: List[str] = []
    lines.append("=== NEAREST NEIGHBORS (TOP 5) ===")
    lines.append(f"Diffusion embedding: {diffusion_path.as_posix()}")
    lines.append(f"CBOW embedding:      {cbow_path.as_posix()}")

    for note in diffusion_notes + cbow_notes:
        lines.append(note)

    if diffusion_notes or cbow_notes:
        lines.append("")

    overlaps: List[Tuple[str, float]] = []
    missing_diffusion: List[str] = []
    missing_cbow: List[str] = []

    for query in queries:
        has_diff = query in diff_index
        has_cbow = query in cbow_index

        if not has_diff and not has_cbow:
            # Per request: skip silently when missing from both.
            continue

        diff_neighbors = top_k_neighbors(query, diff_words, diff_vecs, diff_index, 5) if has_diff else []
        cbow_neighbors = top_k_neighbors(query, cbow_words, cbow_vecs, cbow_index, 5) if has_cbow else []

        if not has_diff:
            missing_diffusion.append(query)
        if not has_cbow:
            missing_cbow.append(query)

        lines.append(f"Query: {query}")
        lines.append("  Diffusion                                CBOW")

        if not has_diff:
            lines.append("  NOTE: missing in Diffusion vocabulary")
        if not has_cbow:
            lines.append("  NOTE: missing in CBOW vocabulary")

        for rank in range(1, 6):
            left = format_neighbor(diff_neighbors, rank)
            right = format_neighbor(cbow_neighbors, rank)
            lines.append(f"  {left:<40} {right}")
        lines.append("")

        if has_diff and has_cbow:
            dset = {w for w, _ in diff_neighbors}
            cset = {w for w, _ in cbow_neighbors}
            overlap = (len(dset & cset) / 5.0) * 100.0
            overlaps.append((query, overlap))

    lines.append("=== OVERLAP ANALYSIS ===")
    if overlaps:
        for query, overlap in overlaps:
            lines.append(f"{query}: {overlap:.1f}%")
        avg = float(np.mean([x for _, x in overlaps]))
        lines.append(f"Average overlap: {avg:.1f}%")
    else:
        lines.append("No overlap scores computed (no shared query words present in both vocabularies).")

    if missing_diffusion:
        lines.append("")
        lines.append("Missing in Diffusion vocab:")
        lines.append(", ".join(sorted(set(missing_diffusion))))

    if missing_cbow:
        lines.append("")
        lines.append("Missing in CBOW vocab:")
        lines.append(", ".join(sorted(set(missing_cbow))))

    output_text = "\n".join(lines)
    print(output_text)

    out_path = Path("results/nearest_neighbors.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output_text + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
