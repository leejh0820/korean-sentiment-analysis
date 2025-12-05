from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from src.data_utils import load_kr3_dataframe, train_test_split_kr3


def run_eda(df: pd.DataFrame) -> None:
    print("==== Basic Info ====")
    print(df.info())
    print()

    print("==== 첫 5개 샘플 ====")
    print(df[["text", "label"]].head())
    print()

    print("==== 레이블 분포 ====")
    print(df["label"].value_counts())
    print()

    print("==== 텍스트 길이 통계 (clean_text 기준) ====")
    lengths = df["clean_text"].str.len()
    print(lengths.describe())
    print()


def build_tfidf_pipeline() -> Pipeline:

    pipe = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20_000,
                    ngram_range=(1, 2),
                    min_df=5,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipe


def main():

    df = load_kr3_dataframe(n_samples=50_000)

    run_eda(df)

    train_df, test_df = train_test_split_kr3(df)

    X_train = train_df["clean_text"].tolist()
    y_train = train_df["label"].astype(int).tolist()
    X_test = test_df["clean_text"].tolist()
    y_test = test_df["label"].astype(int).tolist()

    pipe = build_tfidf_pipeline()

    print("==> Fitting TF-IDF + Logistic Regression...")
    pipe.fit(X_train, y_train)

    print("==> Evaluating...")
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[TF-IDF + LogReg] Test accuracy: {acc:.4f}")
    print("\n[Classification report]\n")
    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
