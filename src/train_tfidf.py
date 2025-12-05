from __future__ import annotations

import os

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.pipeline import Pipeline

from .data_utils import load_kr3_dataframe, train_test_split_kr3
from .plot_utils import plot_confusion_matrix


def prepare_data(n_samples: int | None = None):
    df = load_kr3_dataframe(n_samples=n_samples)
    X_train, y_train, X_test, y_test = train_test_split_kr3(df)
    return X_train, y_train, X_test, y_test


def build_pipeline() -> Pipeline:

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

    X_train, y_train, X_test, y_test = prepare_data(n_samples=50_000)

    pipe = build_pipeline()

    print("==> Fitting TF-IDF + Logistic Regression...")
    pipe.fit(X_train, y_train)

    print("==> Evaluating...")
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[TF-IDF + LogReg] Test accuracy: {acc:.4f}")
    print("\n[Classification report]\n")
    print(classification_report(y_test, y_pred, digits=4))

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "tfidf_logreg.joblib")
    dump(pipe, model_path)
    print(f"\nSaved TF-IDF + LogReg model to: {model_path}")

    os.makedirs(os.path.join("reports", "figures"), exist_ok=True)
    cm_path = os.path.join("reports", "figures", "cm_tfidf.png")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        # labels=["0", "1"],
        title="TF-IDF + LogReg Confusion Matrix",
        out_path=cm_path,
    )
    print(f"Saved confusion matrix figure to: {cm_path}")


if __name__ == "__main__":
    main()
