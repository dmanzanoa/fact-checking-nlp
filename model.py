"""
model.py
========

This module defines a simple recurrent neural network (RNN) for multi class
classification of concatenated claim–evidence texts.  The model uses an
embedding layer followed by a vanilla RNN and a linear classifier.  It also
provides helper functions for preparing sequences and training/testing loops.

The implementation here is deliberately minimal and does not attempt to
compete with state‑of‑the‑art models (e.g. Transformers).  It serves as a
baseline to demonstrate how evidence retrieval can be combined with neural
classification in a single pipeline.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
from torch.utils.data import DataLoader
from torch import nn

__all__ = [
    "SimpleRNN",
    "collate_train",
    "collate_inference",
    "train_epoch",
    "evaluate",
]


class SimpleRNN(nn.Module):
    """A basic character/word embedding + RNN + linear classifier network.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    embedding_dim : int
        Dimensionality of the embedding vectors.
    hidden_dim : int
        Number of hidden units in the RNN.
    num_layers : int
        Number of stacked RNN layers.
    padding_idx : int
        Index in the vocabulary used for padding.  This index will be masked
        out by the embedding layer.
    num_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        padding_idx: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        # Initialise hidden state with zeros on the same device as the input.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        output, hidden = self.rnn(embedded, h0)
        last_hidden = hidden[-1]
        logits = self.fc(last_hidden)
        return logits



def collate_train(
    batch: Iterable[Tuple[str, int]],
    vocab,
    max_len: int,
    padding_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for training.

    Parameters
    ----------
    batch : iterable of (text, label)
        Each element contains a raw text string and its integer label.

    vocab : torchtext.vocab.Vocab
        Vocabulary mapping tokens to indices.

    max_len : int
        Maximum sequence length (longer sequences are truncated, shorter
        sequences are padded).

    padding_index : int
        Index used for padding.

    Returns
    -------
    tuple
        A tuple ``(X, y)`` where ``X`` is a batch of padded sequences and
        ``y`` is the tensor of labels.
    """
    texts, labels = zip(*batch)
    # Tokenise and convert to indices.
    seqs = [vocab(text.split()) for text in texts]
    padded = []
    for seq in seqs:
        seq = seq[:max_len]
        seq += [padding_index] * (max_len - len(seq))
        padded.append(torch.tensor(seq, dtype=torch.long))
    X = torch.stack(padded)
    y = torch.tensor(labels, dtype=torch.long).view(-1, 1)
    return X, y.squeeze()



def collate_inference(
    batch: Iterable[str],
    vocab,
    max_len: int,
    padding_index: int,
) -> torch.Tensor:
    """Collate function for inference (no labels).

    Parameters
    ----------
    batch : iterable of str
        Raw text strings.

    Returns
    -------
    torch.Tensor
        A batch of padded sequences.
    """
    seqs = [vocab(text.split()) for text in batch]
    padded = []
    for seq in seqs:
        seq = seq[:max_len]
        seq += [padding_index] * (max_len - len(seq))
        padded.append(torch.tensor(seq, dtype=torch.long))
    return torch.stack(padded)



def train_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train the model for one epoch and return the average loss.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Loader providing training batches.

    model : torch.nn.Module
        The model to train.

    loss_fn : torch.nn.Module
        Loss function (e.g. ``nn.CrossEntropyLoss``).

    optimizer : torch.optim.Optimizer
        Optimiser for updating model parameters.

    device : str
        Device identifier (e.g. ``"cpu"`` or ``"cuda"``).

    Returns
    -------
    float
        Mean loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(dataloader.dataset)



def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate the model on a validation set.

    Returns
    -------
    tuple
        A pair ``(loss, accuracy)``.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total
