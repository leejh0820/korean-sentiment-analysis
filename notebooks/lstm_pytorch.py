from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


from src.data_utils import (
    load_kr3_dataframe,
    train_test_split_kr3,
    build_vocab,
    KR3TextDataset,
)
from src.models import LSTMClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_lstm_data(
    n_samples: int = 50_000,
    max_len: int = 100,
    max_vocab_size: int = 20_000,
    min_freq: int = 2,
    batch_size: int = 64,
):
    df = load_kr3_dataframe(n_samples=n_samples)
    train_df, test_df = train_test_split_kr3(df)

    stoi, itos = build_vocab(
        train_df["clean_text"].tolist(),
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
    )
    vocab_size = len(stoi)
    print("Vocab size:", vocab_size)

    train_dataset = KR3TextDataset(train_df, stoi=stoi, max_len=max_len)
    test_dataset = KR3TextDataset(test_df, stoi=stoi, max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, vocab_size


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
) -> Tuple[float, float]:

    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
) -> Tuple[float, float]:

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            batch_size = x.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def main():
    # Hyperparameters
    N_SAMPLES = 50_000
    MAX_LEN = 100
    MAX_VOCAB_SIZE = 20_000
    MIN_FREQ = 2
    BATCH_SIZE = 64
    EPOCHS = 3
    LR = 1e-3

    train_loader, test_loader, vocab_size = prepare_lstm_data(
        n_samples=N_SAMPLES,
        max_len=MAX_LEN,
        max_vocab_size=MAX_VOCAB_SIZE,
        min_freq=MIN_FREQ,
        batch_size=BATCH_SIZE,
    )

    model = LSTMClassifier(vocab_size=vocab_size).to(DEVICE)
    print(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            save_path = os.path.join("models", "lstm_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path} (val_acc={val_acc:.4f})")


if __name__ == "__main__":
    main()
