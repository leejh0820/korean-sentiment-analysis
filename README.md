# Korean Restaurant Review Sentiment Classification (KR3)

End-to-end sentiment classification on **Korean restaurant reviews**, comparing:

- **TF-IDF + Logistic Regression** (traditional ML baseline)
- **BiLSTM (PyTorch)** (sequence model)
- **KLUE-BERT base** fine-tuning (pretrained Transformer)

> Training was done on **Google Colab (T4 GPU)**.  
> This repository organizes the code, notebooks, and results in a clean project structure.

---

## 📚 Dataset – KR3

We utilize the [KR3](https://huggingface.co/datasets/leey4n/KR3) dataset for this project:

- **Domain:** Korean restaurant / food-place reviews
- **Labels:**
  - `0` – Negative
  - `1` – Positive
  - (`2` – Ambiguous, filtered out for binary classification)
- **Sampling:**
  - ~50,000 reviews for TF-IDF and LSTM experiments
  - ~30,000 reviews for BERT fine-tuning (due to resource constraints)

The data is loaded directly from Hugging Face via `datasets.load_dataset`, ensuring no raw data files are committed to this repository.

---

## ✨ Methods Implemented

### 1. TF-IDF + Logistic Regression

- **Vectorization:** 1–2-gram TF-IDF (max 20,000 features)
- **Classifier:** Logistic Regression (`max_iter=300`)
- **Script:** `src/train_tfidf.py`

### 2. BiLSTM (PyTorch)

- **Tokenization:** Simple whitespace tokenization
- **Embedding:** Learned embedding layer (20,000-word vocabulary, `min_freq=2`)
- **Architecture:** BiLSTM (Bidirectional LSTM) with dropout and a linear classifier.
- **Script:** `src/train_lstm.py`

### 3. KLUE-BERT base (Transformers)

- **Model:** `klue/bert-base` (pre-trained on Korean text)
- **Process:** Fine-tuned for sequence classification
- **Optimization:** AdamW optimizer with linear warmup/decay scheduler

---

## 🏆 Final Results and Analysis

A comparative analysis of the three models on their respective test sets:

| Model                    | Test Accuracy | Neg. Recall (Class 0) | F1-Score (Macro) | Notes                                                         |
| :----------------------- | :------------ | :-------------------- | :--------------- | :------------------------------------------------------------ |
| **KLUE-BERT base**       | **0.9623**    | **0.8579**            | **0.9298**       | **Best overall**; superior context modeling is crucial.       |
| **BiLSTM**               | 0.9246        | _N/A_                 | _N/A_            | Strong sequence baseline; limited by lack of pre-training.    |
| **TF-IDF + LogisticReg** | 0.9237        | **0.5675**            | 0.8268           | Simple, fast, but fails drastically on subtle/negative cases. |

### Key Takeaway

The project demonstrates that **context-aware models (BERT)** are essential for robust Korean sentiment analysis. While traditional methods achieved high overall accuracy (due to dataset imbalance), the TF-IDF model's extremely low **Negative Recall (0.5675)** shows its failure to capture the nuance of the minority class, a weakness effectively overcome by the deep contextual understanding of **KLUE-BERT**.

#### Example BERT Classification Report:

```text
# (This report is based on the best validation accuracy)

              precision    recall  f1-score   support

           0     0.9047    0.8579    0.8807       931
           1     0.9742    0.9836    0.9789      5069

    accuracy                         0.9623      6000
   macro avg     0.9395    0.9208    0.9298      6000
weighted avg     0.9623    0.9623    0.9621      6000
```
