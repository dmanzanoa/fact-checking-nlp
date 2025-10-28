"""
main.py
=======

This script provides an example of how to tie together the modules in this
repository to build a simple fact checking system.  It demonstrates:

1. Loading and preprocessing claim/evidence datasets.
2. Retrieving the top evidence passages for each claim using either TF‑IDF or
   BM25.
3. Building a vocabulary over the combined claim–evidence texts and training
   a simple RNN classifier to predict the veracity of claims.

The script is intentionally minimalist and is intended as a starting point
for further experimentation.  You can substitute other retrieval models or
classifiers by following the same interface.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from data_processing import load_datasets
from retrieval import retrieve_top_evidence_tfidf, retrieve_top_evidence_bm25
from model import SimpleRNN, collate_train, collate_inference, train_epoch, evaluate


def build_vocab(texts: Iterable[str]) -> torchtext.vocab.Vocab:
    """Build a torchtext vocabulary from an iterable of tokenised strings."""
    def yield_tokens(data_iter: Iterable[str]):
        for text in data_iter:
            yield text.split()

    vocab = build_vocab_from_iterator(yield_tokens(texts), specials=("<unk>", "<pad>"))
    vocab.set_default_index(vocab["<unk>"])
    return vocab


def run_training(
    use_bm25: bool = True,
    k: int = 5,
    epochs: int = 3,
    batch_size: int = 16,
) -> None:
    """End‑to‑end training example.

    Parameters
    ----------
    use_bm25 : bool, default ``True``
        Whether to use BM25 (if available) for evidence retrieval.  If False,
        TF‑IDF retrieval is used instead.

    k : int, default ``5``
        Number of evidence passages to retrieve per claim.

    epochs : int, default ``3``
        Number of epochs to train the RNN classifier.

    batch_size : int, default ``16``
        Batch size for training and evaluation.
    """
    # Load datasets.
    datasets = load_datasets("data")

    # Retrieve evidence for training claims.
    if use_bm25:
        texts, _ = retrieve_top_evidence_bm25(
            datasets.claims_train, datasets.evidences, k=k
        )
    else:
        texts, _ = retrieve_top_evidence_tfidf(
            datasets.claims_train, datasets.evidences, k=k
        )

    # Build vocabulary over the concatenated claim–evidence strings.
    vocab = build_vocab(texts)
    padding_index = vocab["<pad>"]

    # Map labels to integer indices.
    label_encoder = {lbl: idx for idx, lbl in enumerate(sorted(datasets.claims_train["claim_label"].unique()))}
    labels = datasets.claims_train["claim_label"].map(label_encoder).tolist()

    # Create DataLoader for training.
    dataset = list(zip(texts, labels))
    collate_fn = lambda batch: collate_train(batch, vocab, max_len=30, padding_index=padding_index)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Instantiate model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleRNN(
        vocab_size=len(vocab),
        embedding_dim=50,
        hidden_dim=64,
        num_layers=1,
        padding_idx=padding_index,
        num_classes=len(label_encoder),
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train loop.
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(dataloader, model, loss_fn, optimizer, device)
        print(f"Epoch {epoch}: train loss = {train_loss:.4f}")

    print("Training complete.  You can now apply the model to unseen claims using the inference utilities.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a fact‟checking experiment")
    parser.add_argument("--bm25", action="store_true", help="Use BM25 retrieval instead of TF‑IDF")
    parser.add_argument("--k", type=int, default=5, help="Number of evidence passages to retrieve per claim")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs for training")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    args = parser.parse_args()
    run_training(use_bm25=args.bm25, k=args.k, epochs=args.epochs, batch_size=args.batch_size)
