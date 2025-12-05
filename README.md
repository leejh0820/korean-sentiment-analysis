# Korean Restaurant Review Sentiment Classification (KR3)

End-to-end **sentiment classification** on Korean restaurant reviews, comparing:

- **TF–IDF + Logistic Regression** (traditional ML baseline)  
- **BiLSTM (PyTorch)** (sequence model)  
- **KLUE-BERT base fine-tuning** (pretrained Transformer)

Training was done on **Google Colab (T4 GPU)**.  
This repository contains the code, notebooks, and results in a clean, reproducible project structure.

---

## 📚 Dataset – KR3

We utilize the **KR3** dataset for this project:

- **Domain**: Korean restaurant / food-place reviews  
- **Labels**:
  - `0` – Negative  
  - `1` – Positive  
  - `2` – Ambiguous (*filtered out for binary classification*)  
- **Sampling**:
  - ~50,000 reviews for TF–IDF and BiLSTM experiments  
  - ~30,000 reviews for BERT fine-tuning (due to resource constraints)  

The data is loaded directly from Hugging Face via `datasets.load_dataset`,  
so **no raw data files are committed** to this repository.

---

## ✨ Methods Implemented

### 1. TF–IDF + Logistic Regression

- **Vectorization**: 1–2-gram TF–IDF (max 20,000 features)  
- **Classifier**: Logistic Regression (`max_iter=300`)  
- **Script**: `src/train_tfidf.py`  

This serves as a strong yet lightweight **traditional ML baseline**.

---

### 2. BiLSTM (PyTorch)

- **Tokenization**: Simple whitespace tokenization  
- **Embedding**: Learned embedding layer  
  - Vocabulary size: 20,000  
  - `min_freq = 2`  
- **Architecture**:
  - Bidirectional LSTM (BiLSTM)  
  - Dropout regularization  
  - Final linear classifier layer  
- **Script**: `src/train_lstm.py`  

This model captures **sequential information** in reviews without any pretraining.

---

### 3. KLUE-BERT base (Transformers)

- **Model**: `klue/bert-base` (pre-trained on Korean text)  
- **Task**: Sequence classification (binary sentiment)  
- **Optimization**:
  - AdamW optimizer  
  - Linear learning-rate scheduler with warmup/decay  
- **Script**: `src/train_bert.py`  

This is the main **context-aware** model in the project.

---

## 🏋️‍♂️ Training Setup

- Environment: **Google Colab** (T4 GPU)  
- Frameworks:
  - PyTorch / TorchText (BiLSTM)  
  - Hugging Face `transformers`, `datasets` (KLUE-BERT)  
  - Scikit-learn (TF–IDF + Logistic Regression)

---

## 🏆 Final Results and Analysis

A comparative analysis of the three models on their respective test sets:

| Model                  | Test Accuracy | Neg. Recall (Class 0) | F1-Score (Macro) | Notes                                                     |
| ---------------------- | ------------: | ---------------------: | ----------------:| ---------------------------------------------------------- |
| **KLUE-BERT base**     | **0.9623**    | **0.8579**             | **0.9298**       | Best overall; context modeling is crucial.                |
| **BiLSTM**             | 0.9246        | N/A                    | N/A              | Strong sequence baseline; limited by lack of pretraining. |
| **TF–IDF + LogReg**    | 0.9237        | 0.5675                 | 0.8268           | Fast and simple, but struggles on subtle/negative cases.  |

### Key Takeaway

The project demonstrates that **context-aware models (BERT)** are essential for robust **Korean sentiment analysis**.

While traditional methods achieve high overall accuracy (partly due to dataset imbalance), the TF–IDF model’s extremely low **Negative Recall** (`0.5675`) reveals its weakness on the **minority class**.  
In contrast, **KLUE-BERT** maintains high overall accuracy **and** high recall on negative reviews, thanks to its deep contextual understanding.

---

## 📊 KLUE-BERT Base – Final Classification Report

> Actual result from the best validation checkpoint:

```text
              precision    recall  f1-score   support

           0     0.9047    0.8579    0.8807       931
           1     0.9742    0.9836    0.9789      5069

    accuracy                         0.9623      6000
   macro avg     0.9395    0.9208    0.9298      6000
weighted avg     0.9623    0.9623    0.9621      6000

