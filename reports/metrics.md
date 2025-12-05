# Metrics Summary

## Accuracy Comparison

| Model                      | Split | Accuracy   |
| -------------------------- | ----- | ---------- |
| TF-IDF + LogisticReg       | Test  | **0.9237** |
| BiLSTM                     | Val   | **0.9246** |
| KLUE-BERT base (fine-tune) | Test  | **0.9565** |

- TF-IDF + Logistic Regression: 간단한 sparse 특징에도 불구하고 꽤 높은 정확도를 달성함.
- BiLSTM: 시퀀스 정보를 활용해 TF-IDF 대비 소폭 향상된 성능을 보임(Validation 기준).
- KLUE-BERT: 사전학습된 한국어 BERT를 파인튜닝하여 가장 높은 성능을 달성함.

---

## KLUE-BERT Classification Report (Test)

```text
accuracy = 0.9565

              precision    recall  f1-score   support

           0     0.8588    0.8579    0.8584       922
           1     0.9742    0.9744    0.9743      5078

    accuracy                         0.9565      6000
   macro avg     0.9165    0.9162    0.9163      6000
weighted avg     0.9565    0.9565    0.9565      6000


## TF-IDF + Logistic Regression Classification Report (Test)

Test accuracy: 0.9237

              precision    recall  f1-score   support

           0     0.9043    0.5675    0.6973      1549
           1     0.9258    0.9890    0.9563      8451

    accuracy                         0.9237     10000
   macro avg     0.9151    0.7782    0.8268     10000
weighted avg     0.9225    0.9237    0.9162     10000
```
