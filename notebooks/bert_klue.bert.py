from __future__ import annotations

import os
from typing import Tuple

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data_utils import (
    load_kr3_dataframe,
    train_test_split_kr3,
    BertDataset,
    bert_collate_fn,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_bert_data(
    n_samples: int = 30_000,
    max_len: int = 128,
    batch_size: int = 32,
):
    df = load_kr3_dataframe(n_samples=n_samples)
    train_df, test_df = train_test_split_kr3(df)

    model_name = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = BertDataset(train_df, tokenizer=tokenizer, max_len=max_len)
    test_dataset = BertDataset(test_df, tokenizer=tokenizer, max_len=max_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=bert_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=bert_collate_fn,
    )

    return train_loader, test_loader, tokenizer


def train_one_epoch_bert(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Train BERT", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_size = batch["input_ids"].size(0)
        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(batch["labels"].detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def eval_bert(
    model: torch.nn.Module,
    loader: DataLoader,
) -> Tuple[float, float, str]:

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval BERT", leave=False):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            logits = outputs.logits

            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(batch["labels"].detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    return avg_loss, acc, report


def main():
    # Hyperparameters
    N_SAMPLES = 30_000
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 3
    LR = 2e-5
    MODEL_NAME = "klue/bert-base"

    # 1) Data
    train_loader, test_loader, tokenizer = prepare_bert_data(
        n_samples=N_SAMPLES,
        max_len=MAX_LEN,
        batch_size=BATCH_SIZE,
    )

    # 2) Model, optimizer, scheduler
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    num_training_steps = EPOCHS * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_val_acc = 0.0
    best_report = ""

    # 3) Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch_bert(
            model,
            train_loader,
            optimizer,
            scheduler,
        )
        val_loss, val_acc, report = eval_bert(model, test_loader)

        print(
            f"[BERT Epoch {epoch}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_report = report
            os.makedirs("models", exist_ok=True)
            save_path = os.path.join("models", "bert_klue_base.pt")
            torch.save(model.state_dict(), save_path)
            print(
                f"  -> New best BERT model saved to {save_path} (val_acc={val_acc:.4f})"
            )

    print("\n[BERT classification report]\n")
    print(best_report)


if __name__ == "__main__":
    main()
