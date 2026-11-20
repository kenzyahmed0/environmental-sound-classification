# Ablation Study

## 1. Objective

This ablation study investigates the effect of applying StandardScaler to the YAMNet + SVM model for environmental sound classification.

Two configurations were compared while keeping the dataset, train-validation split, SVM algorithm, and hyperparameters unchanged.

## 2. Experimental Setup

- Dataset: UrbanSound8K
- Feature extractor: Pretrained YAMNet
- Training samples: 5,435
- Validation samples: 838
- SVM kernel: RBF
- C: 10
- Gamma: scale
- Class weight: balanced

The only changed component was the use of StandardScaler before the SVM classifier.

## 3. Results

| Configuration | Validation Accuracy | Validation Macro F1 | Training Time (seconds) |
|---|---:|---:|---:|
| With StandardScaler | 81.03% | 81.24% | 1.30 |
| Without StandardScaler | 80.67% | 80.49% | 1.08 |

## 4. Analysis

The configuration using StandardScaler achieved better validation performance.

- Accuracy improved by approximately 0.36 percentage points.
- Macro F1 improved by approximately 0.75 percentage points.
- Training time increased slightly by approximately 0.22 seconds.

The improvement in Macro F1 suggests that feature scaling helped the SVM achieve better overall class-level performance.

## 5. Conclusion

StandardScaler had a positive effect on the validation performance of the YAMNet + SVM model.

Therefore, StandardScaler was retained in the final model pipeline.

The experiment used only the training and validation sets. The test set was not used for selecting this component, avoiding test-data leakage.

## 6. Limitations

This experiment evaluates the effect of one component in the selected model pipeline. The results do not establish that scaling will improve every model or dataset. Additional experiments with different random seeds would help assess the consistency of the observed improvement.

## 7. Reproducibility

The experiment was implemented in:

`experiments/ablation/run_ablation.py`

The numerical results were saved in:

`experiments/results/ablation_scaling.csv`
