#!/usr/bin/env python3
"""
NNLM baseline: Bengio et al. (2003) feedforward neural language model.
Trains on the same tokenized corpus and vocab as the diffusion method.
Exports word embeddings to JSON in the same format as diffusion checkpoints.

Usage:
  python train_nnlm.py
  python train_nnlm.py --context 4 --hidden 128 --dim 32 --epochs 50
  python train_nnlm.py --run-eval

Architecture:
  Input: n-gram context (n-1 words) -> embedding lookup -> concat -> tanh hidden -> softmax
  Embedding layer weights become the exported word vectors.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
except ImportError:
    raise SystemExit(
        "Dependencies are required. Install with:\n"
        "  pip install torch tqdm\n"
    )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_ngrams(
    sentences: List[List[str]],
    vocab_set: set,
    context_size: int,
) -> List[List[int]]:
    """Build (context_size) → target n-gram index lists. Words not in vocab are skipped."""
    ngrams: List[List[int]] = []
    # word_to_idx will be built by caller; here we just return token lists
    # (actual indexing happens after vocab is finalized)
    return ngrams


class NgramDataset(Dataset):
    def __init__(
        self,
        sentences: List[List[str]],
        word_to_idx: Dict[str, int],
        context_size: int,
    ):
        self.data: List[tuple] = []
        for sent in tqdm(sentences, desc="Building n-grams", unit="sent", leave=False):
            indices = [word_to_idx[w] for w in sent if w in word_to_idx]
            for i in range(context_size, len(indices)):
                context = indices[i - context_size: i]
                target = indices[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NNLM(nn.Module):
    """Bengio et al. 2003 feedforward neural language model."""

    def __init__(self, vocab_size: int, embed_dim: int, context_size: int, hidden_size: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.hidden = nn.Linear(context_size * embed_dim, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.tanh = nn.Tanh()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: (batch, context_size)
        embeds = self.embeddings(context)           # (batch, context_size, embed_dim)
        embeds = embeds.view(embeds.size(0), -1)   # (batch, context_size * embed_dim)
        hidden = self.tanh(self.hidden(embeds))     # (batch, hidden_size)
        logits = self.output(hidden)                # (batch, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: NNLM,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch", leave=False)
    for context, target in pbar:
        context, target = context.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(context)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        batch_n = context.size(0)
        total_loss += loss.item() * batch_n
        seen += batch_n
        running_loss = total_loss / max(seen, 1)
        pbar.set_postfix(loss=f"{running_loss:.4f}")
    return total_loss / len(loader.dataset)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_embeddings(
    model: NNLM,
    vocab_list: List[str],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weights = model.embeddings.weight.detach().cpu().numpy().astype(np.float32)
    vectors = {word: weights[i].tolist() for i, word in enumerate(vocab_list)}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(vectors, f)
    print(f"Saved embeddings: {out_path}  ({len(vectors)} words, dim={weights.shape[1]})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train NNLM baseline (Bengio et al. 2003)")
    ap.add_argument("--tokenized", default="data/fairytales_tokenized.json")
    ap.add_argument("--tfidf", default="data/fairytales_word_tf-idfs.json",
                    help="Used only to fix vocab order (same as diffusion)")
    ap.add_argument("--context", type=int, default=4, help="N-gram context size (n-1 words)")
    ap.add_argument("--dim", type=int, default=32, help="Embedding dimension")
    ap.add_argument("--hidden", type=int, default=128, help="Hidden layer size")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="", help="Output JSON path (auto-named if empty)")
    ap.add_argument("--run-eval", action="store_true", help="Run evaluate_analogies.py after export")
    ap.add_argument("--questions", default="evaluation/analogy_questions.json")
    ap.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = ap.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if cuda_available else "cpu"
    elif device_str.startswith("cuda") and not cuda_available:
        raise SystemExit("Requested CUDA device, but torch.cuda.is_available() is False.")

    device = torch.device(device_str)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(idx)
        print(f"Using device: cuda:{idx} ({gpu_name})")
    else:
        print("Using device: cpu")

    # Load data
    tokenized_path = Path(args.tokenized)
    tfidf_path = Path(args.tfidf)
    sentences: List[List[str]] = load_json(tokenized_path)
    tf_idfs: Dict = load_json(tfidf_path)

    # Vocab: use tf_idfs.keys() to match diffusion vocab exactly
    vocab_list = list(tf_idfs.keys())
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}
    vocab_size = len(vocab_list)

    print(f"Vocab size: {vocab_size}")
    print(f"Context size: {args.context}  Dim: {args.dim}  Hidden: {args.hidden}")
    print(f"Epochs: {args.epochs}  Batch: {args.batch}  LR: {args.lr}")

    # Dataset
    dataset = NgramDataset(sentences, word_to_idx, args.context)
    print(f"N-gram examples: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=0, pin_memory=(device_str == "cuda"))

    # Model
    model = NNLM(vocab_size, args.dim, args.context, args.hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Train
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train(model, loader, optimizer, criterion, device, epoch, args.epochs)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  ({time.time()-t0:.1f}s)")

    print(f"Training done in {time.time()-t0:.1f}s")

    # Export
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("data/nnlm") / f"nnlm_vectors_{args.dim}d_ctx{args.context}.json"

    export_embeddings(model, vocab_list, out_path)

    # Optionally evaluate
    if args.run_eval:
        import subprocess, sys as _sys
        evaluator = "evaluation/evaluate_google_analogies.py" if args.questions.lower().endswith(".txt") else "evaluation/evaluate_analogies.py"
        cmd = [
            _sys.executable, evaluator,
            "--embeddings", str(out_path),
            "--questions", args.questions,
        ]
        print("\nRunning evaluator:\n  " + " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
