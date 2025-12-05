from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import re
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from collections import Counter
from typing import Optional, Tuple, Iterable, Dict, Sequence, Union

import torch
from torch.utils.data import Dataset


def load_kr3_raw() -> pd.DataFrame:
    ds = load_dataset("leey4n/KR3", split="train")

    if "__index_level_0__" in ds.column_names:
        ds = ds.remove_columns(["__index_level_0__"])

    df = ds.to_pandas()
    return df


def load_kr3_binary(
    sample_size: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = load_kr3_raw()

    df = df[df["Rating"] != 2].copy()

    df = df.rename(columns={"Review": "text", "Rating": "label"})

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def clean_korean_text(text: str) -> str:

    text = str(text)
    text = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_clean_text(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["clean_text"] = df["text"].map(clean_korean_text)
    df = df[df["clean_text"].str.len() > 0]
    return df.reset_index(drop=True)


def prepare_kr3(
    sample_size: Optional[int] = 50_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_df, test_df = load_kr3_binary(
        sample_size=sample_size,
        test_size=test_size,
        random_state=random_state,
    )
    train_df = add_clean_text(train_df)
    test_df = add_clean_text(test_df)
    return train_df, test_df


def load_kr3_dataframe(
    n_samples: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:

    df = load_kr3_raw()

    df = df[df["Rating"] != 2].copy()

    df = df.rename(columns={"Review": "text", "Rating": "label"})

    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_state)

    df = add_clean_text(df)

    return df.reset_index(drop=True)


def train_test_split_kr3(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, y_train, X_test, y_test


def build_vocab(
    texts: Iterable[str],
    min_freq: int = 2,
    max_size: Optional[int] = None,
    specials: Sequence[str] = ("<pad>", "<unk>"),
    **kwargs,
) -> Tuple[Dict[str, int], Sequence[str]]:

    counter: Counter = Counter()

    for text in texts:
        text = str(text)
        tokens = text.split()
        counter.update(tokens)

    sorted_tokens = [tok for tok, freq in counter.most_common() if freq >= min_freq]

    if max_size is not None:
        sorted_tokens = sorted_tokens[:max_size]

    stoi: Dict[str, int] = {}

    for sp in specials:
        if sp not in stoi:
            stoi[sp] = len(stoi)

    for tok in sorted_tokens:
        if tok not in stoi:
            stoi[tok] = len(stoi)

    itos = [None] * len(stoi)
    for token, idx in stoi.items():
        itos[idx] = token

    return stoi, itos


class KR3TextDataset(Dataset):

    def __init__(
        self,
        data: Union[pd.DataFrame, Sequence[str]],
        labels: Optional[Sequence[int]] = None,
        stoi: Optional[Dict[str, int]] = None,
        vocab: Optional[Dict[str, int]] = None,
        max_len: int = 128,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        text_col: str = "clean_text",
        label_col: str = "label",
    ) -> None:
        # vocab / stoi 통합 처리
        if vocab is None:
            vocab = stoi
        if vocab is None:
            raise ValueError(
                "KR3TextDataset: vocab 또는 stoi 중 하나는 반드시 필요합니다."
            )

        self.vocab = vocab
        self.max_len = max_len

        self.pad_idx = vocab.get(pad_token, 0)
        self.unk_idx = vocab.get(unk_token, vocab.get("<unk>", 1))

        if isinstance(data, pd.DataFrame):
            df = data

            self.texts = df[text_col].astype(str).tolist()
            self.labels = df[label_col].astype(int).tolist()

        else:
            if labels is None:
                raise ValueError(
                    "KR3TextDataset: texts 시퀀스를 직접 넘길 때는 labels도 함께 넘겨야 합니다."
                )
            self.texts = [str(t) for t in data]
            self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def _encode(self, text: str) -> torch.Tensor:
        tokens = text.split()
        ids = []
        for tok in tokens[: self.max_len]:
            ids.append(self.vocab.get(tok, self.unk_idx))

        if len(ids) < self.max_len:
            ids = ids + [self.pad_idx] * (self.max_len - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = self._encode(text)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return input_ids, label_tensor


def train_test_split_kr3_df(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


class BertDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        labels: Optional[Sequence[int]] = None,
        max_len: int = 128,
        text_col: str = "clean_text",
        label_col: str = "label",
    ) -> None:

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.texts = data[text_col].astype(str).tolist()
        if labels is None:
            self.labels = data[label_col].astype(int).tolist()
        else:
            self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def bert_collate_fn(batch):

    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    if "token_type_ids" in batch[0]:
        collated["token_type_ids"] = torch.stack(
            [item["token_type_ids"] for item in batch]
        )

    return collated
