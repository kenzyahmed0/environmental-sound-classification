
# Computational Cost Analysis

## 1. Hardware Specifications

All experiments were conducted on the following machine:

| Component | Specification |
|---|---|
| CPU | 13th Gen Intel Core i7-13650HX |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) |
| Integrated GPU | Intel UHD Graphics |
| RAM | 16 GB |
| Operating System | Windows 11 |

**Important note:** TensorFlow GPU support was unavailable in the native Windows environment. Therefore, TensorFlow-based operations, including CNN inference and YAMNet feature extraction, were executed without native TensorFlow GPU acceleration.

## 2. Model Size

The following table reports the size of each saved model checkpoint.

| Model | Checkpoint Size (MB) |
|---|---:|
| YAMNet + SVM | 16.395 |
| YAMNet + MLP | 3.427 |
| MFCC + SVM | 2.119 |
| CNN | 1.338 |

The checkpoint sizes represent the saved model files. The pretrained YAMNet feature extractor is loaded separately and is not included in these checkpoint sizes.

## 3. Model Parameters and Complexity

| Model | Total Parameters | Trainable Parameters | Support Vectors |
|---|---:|---:|---:|
| YAMNet + SVM | N/A | N/A | 2,058 |
| YAMNet + MLP | 296,586 | 296,586 | N/A |
| MFCC + SVM | N/A | N/A | 2,813 |
| CNN | 111,370 | 110,922 | N/A |

For the SVM models, the number of support vectors is reported instead of neural network parameter counts because SVMs use a different model representation.

## 4. Inference Time

Inference time was measured using repeated predictions and averaged over five runs.

| Model | Average Inference Time (seconds) | Inference Time (ms/sample) |
|---|---:|---:|
| YAMNet + SVM | 6.1097 | 2.4846 |
| YAMNet + MLP | 0.1033 | 0.0420 |
| MFCC + SVM | 1.8910 | 0.7690 |
| CNN | 0.3133 | 9.7895 |

**Measurement limitation:** The sklearn models were evaluated on the complete test feature set (2,459 samples), while the CNN timing was measured on one batch of 32 samples. Therefore, these inference measurements are preliminary and should not be interpreted as a fully controlled end-to-end comparison.

The reported sklearn inference times measure prediction on precomputed features. They do not include audio preprocessing or feature extraction.

## 5. YAMNet Feature Extraction Cost

YAMNet feature extraction was measured separately using 20 test audio samples.

| Measurement | Result |
|---|---:|
| Number of audio samples | 20 |
| Total extraction time | 2.3606 seconds |
| Average time per audio sample | 0.1180 seconds |

This measurement represents the feature extraction stage only, not the complete prediction pipeline.

## 6. Training Time

The original training process did not record reliable per-model training times. Therefore, exact training times are not reported here.

A separate ablation experiment measured the YAMNet + SVM classifier fitting time:

| Configuration | Training Time |
|---|---:|
| With StandardScaler | 1.30 seconds |
| Without StandardScaler | 1.08 seconds |

These values belong only to the ablation experiment and must not be interpreted as the complete original training times.

## 7. Computational Cost Discussion

The results show differences in model size and inference requirements.

- YAMNet + MLP has the lowest measured sklearn prediction latency per sample.
- CNN has the smallest checkpoint size among the evaluated models.
- YAMNet + SVM uses the largest checkpoint file among the four classifiers.
- The CNN contains 111,370 parameters, while YAMNet + MLP contains 296,586 parameters.
- YAMNet feature extraction introduces an additional computational stage before classification.

The comparison should consider both the classifier cost and the cost of extracting features from raw audio. In particular, YAMNet-based models require pretrained feature extraction, while the CNN processes log-Mel spectrogram inputs.

## 8. Limitations

- TensorFlow GPU acceleration was unavailable in the native Windows environment.
- CNN inference was measured on a single batch of 32 samples, unlike the full-test-set measurements for sklearn models.
- Inference measurements for sklearn models exclude feature extraction.
- Exact original training times were not recorded.
- YAMNet feature extraction was measured using 20 samples rather than the entire test set.

Future benchmarking should measure all models using the same test samples, batch protocol, hardware configuration, and end-to-end pipeline.
