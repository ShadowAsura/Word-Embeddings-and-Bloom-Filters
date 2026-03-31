#!/usr/bin/env python3
"""
RNNLM baseline: Recurrent Neural Network Language Model (Mikolov et al. 2010).
Trains on the same tokenized corpus and vocab as the diffusion method.
Exports the RNN input embedding weights to JSON in the same format as diffusion checkpoints.

Usage:
  python train_rnnlm.py
  python train_rnnlm.py --hidden 128 --dim 32 --epochs 50 --rnn-type lstm
  python train_rnnlm.py --run-eval

Architecture:
  Embedding -> RNN/LSTM/GRU -> Linear -> Softmax over vocab
  The embedding layer weights are exported as word vectors.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
except ImportError:
    raise SystemExit("Dependencies are required. Install with:  pip install torch tqdm")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class SentenceDataset(Dataset):
    """Each item is a full sentence as a list of indices (variable length)."""

    def __init__(self, sentences: List[List[str]], word_to_idx: Dict[str, int]):
        self.samples: List[torch.Tensor] = []
        for sent in tqdm(sentences, desc="Building sentence dataset", unit="sent", leave=False):
            indices = [word_to_idx[w] for w in sent if w in word_to_idx]
            if len(indices) >= 2:
                self.samples.append(torch.tensor(indices, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad sentences to the same length within a batch. Returns (inputs, targets, lengths)."""
    lengths = torch.tensor([len(s) - 1 for s in batch], dtype=torch.long)  # predict next token
    max_len = int(lengths.max().item())
    inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    targets = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        L = len(s) - 1
        inputs[i, :L] = s[:-1]
        targets[i, :L] = s[1:]
    return inputs, targets, lengths


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class RNNLM(nn.Module):
    """Recurrent Neural Network Language Model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "rnn",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.rnn_type]
        self.rnn = rnn_cls(
            embed_dim, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor):
        # inputs: (batch, seq_len)
        embeds = self.drop(self.embeddings(inputs))         # (batch, seq_len, embed_dim)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        if self.rnn_type == "lstm":
            out_packed, _ = self.rnn(packed)
        else:
            out_packed, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # (batch, seq_len, hidden)
        out = self.drop(out)
        logits = self.output(out)                           # (batch, seq_len, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: RNNLM,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch", leave=False)
    for inputs, targets, lengths in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(inputs, lengths)     # (batch, seq_len, vocab_size)

        # Flatten; mask padding positions
        batch_size, seq_len, vocab_size = logits.shape
        mask = (targets != 0)  # non-padding positions (padding_idx=0 used for both PAD and first real token — careful!)
        # Use length mask to be safe
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for i, L in enumerate(lengths):
            mask[i, :L] = True

        logits_flat = logits[mask]   # (valid_tokens, vocab_size)
        targets_flat = targets[mask] # (valid_tokens,)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        n = targets_flat.size(0)
        total_loss += loss.item() * n
        total_tokens += n
        running = total_loss / max(total_tokens, 1)
        pbar.set_postfix(loss=f"{running:.4f}")

    return total_loss / max(total_tokens, 1)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_embeddings(model: RNNLM, vocab_list: List[str], out_path: Path) -> None:
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
    ap = argparse.ArgumentParser(description="Train RNNLM baseline (Mikolov et al. 2010)")
    ap.add_argument("--tokenized", default="data/fairytales_tokenized.json")
    ap.add_argument("--tfidf", default="data/fairytales_word_tf-idfs.json",
                    help="Used only to fix vocab order (same as diffusion)")
    ap.add_argument("--dim", type=int, default=32, help="Embedding dimension")
    ap.add_argument("--hidden", type=int, default=128, help="RNN hidden size")
    ap.add_argument("--layers", type=int, default=1, help="Number of RNN layers")
    ap.add_argument("--rnn-type", default="rnn", choices=["rnn", "lstm", "gru"],
                    help="RNN cell type")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
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
    print(f"RNN type: {args.rnn_type.upper()}  Dim: {args.dim}  Hidden: {args.hidden}  Layers: {args.layers}")
    print(f"Epochs: {args.epochs}  Batch: {args.batch}  LR: {args.lr}")

    dataset = SentenceDataset(sentences, word_to_idx)
    print(f"Sentences: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=(device_str == "cuda"),
    )

    model = RNNLM(
        vocab_size=vocab_size,
        embed_dim=args.dim,
        hidden_size=args.hidden,
        num_layers=args.layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, criterion, device, epoch, args.epochs)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  ({time.time()-t0:.1f}s)")

    print(f"Training done in {time.time()-t0:.1f}s")

    # Export
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("data/rnnlm") / f"rnnlm_{args.rnn_type}_vectors_{args.dim}d.json"

    export_embeddings(model, vocab_list, out_path)

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
