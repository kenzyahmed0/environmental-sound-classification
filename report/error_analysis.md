# Error Analysis – EchoMind

## 1. Overview

This section analyzes the test performance of the selected environmental sound classification model, **YAMNet + SVM**, using the official UrbanSound8K test split.

The selected model achieved:

- Test Accuracy: **75.68%**
- Test Macro F1-score: **76.15%**
- Test Weighted F1-score: **75.42%**

The analysis focuses on class-wise performance and possible sources of misclassification.

## 2. Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Air Conditioner | 0.6500 | 0.5200 | 0.5778 |
| Car Horn | 0.7642 | 0.8526 | 0.8060 |
| Children Playing | 0.8509 | 0.7800 | 0.8139 |
| Dog Bark | 0.8215 | 0.8133 | 0.8174 |
| Drilling | 0.8393 | 0.7833 | 0.8103 |
| Engine Idling | 0.6749 | 0.6074 | 0.6394 |
| Gun Shot | 0.7023 | 0.9892 | 0.8214 |
| Jackhammer | 0.6364 | 0.8203 | 0.7167 |
| Siren | 0.8939 | 0.7224 | 0.7991 |
| Street Music | 0.7465 | 0.8933 | 0.8134 |

## 3. Main Observations

### Weakest-performing classes

- **Air Conditioner:** Has the lowest recall (52%), meaning many actual air-conditioner samples were not correctly identified.
- **Engine Idling:** Recall is 60.74%, indicating difficulty recognizing some engine-idling recordings.
- **Jackhammer:** Has the lowest precision (63.64%), suggesting that several samples predicted as jackhammer belong to other classes.

### Stronger-performing classes

- **Gun Shot:** Achieved a very high recall of 98.92%, meaning nearly all actual gun-shot samples were detected. However, its precision is 70.23%, indicating false-positive predictions.
- **Street Music:** Achieved 89.33% recall.
- **Car Horn:** Achieved 85.26% recall.

## 4. Possible Sources of Errors

The observed errors may be related to:

1. **Acoustic similarity:** Some environmental sounds can share frequency patterns and background characteristics.
2. **Background noise:** Recordings may contain overlapping sounds that make classification more difficult.
3. **Class imbalance:** The test set contains 1,325 street-music samples, compared with substantially fewer samples in several other classes. This imbalance can influence overall performance.
4. **Recording variability:** Differences in distance, recording conditions, and sound intensity may affect the extracted audio representations.

These are possible explanations; further inspection of individual misclassified samples would be needed to confirm them.

## 5. Conclusion

YAMNet + SVM provides a reasonable baseline for environmental sound classification, achieving 75.68% test accuracy and 76.15% macro F1-score.

The results show that performance varies across sound categories. Air Conditioner and Engine Idling are among the most challenging classes, while Gun Shot and Street Music have relatively high recall.

Future improvements could investigate class balancing, data augmentation, and more advanced audio classification architectures.