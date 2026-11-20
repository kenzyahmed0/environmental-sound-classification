# EchoMind – Environmental Sound Classification

## 1. Project Title, and Selected Idea

**Project Title:** EchoMind – Environmental Sound Classification

**Selected Idea:** Environmental Sound Classification using Machine Learning and Deep Learning.


## 2. Project Description and Best Results

EchoMind is an environmental sound classification system that identifies sound categories from audio recordings.

The project uses the UrbanSound8K dataset, which contains 8,732 audio recordings belonging to 10 environmental sound classes.

Four approaches were implemented and compared:

1. MFCC + SVM
2. YAMNet + SVM
3. YAMNet + MLP
4. CNN using Log-Mel Spectrograms

**Selected model:** YAMNet + SVM, selected according to validation Macro F1.

- Validation Accuracy: 81.03%
- Validation Macro F1: 81.24%
- Test Accuracy: 75.68%
- Test Macro F1: 76.15%

YAMNet + MLP achieved the highest test Macro F1 of 78.00%. However, YAMNet + SVM remained the selected model because model selection was based on validation performance, not test performance.

## 3. Environment Setup

### Requirements

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Librosa
- SoundFile
- TensorFlow
- TensorFlow Hub
- Joblib
- Gradio
- Jupyter

### Installation

Clone the repository:

```bash
git clone https://github.com/kenzyahmed0/environmental-sound-classification.git
cd environmental-sound-classification
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Dataset Download and Preprocessing

The project uses the **UrbanSound8K** dataset.

Dataset source:
https://urbansounddataset.weebly.com/urbansound8k.html

Download the dataset:

```bash
python -m data.download_dataset
```

Preprocess the dataset and generate the metadata splits:

```bash
python -m data.preprocess
```

The dataset is divided using the official folds:

- Training: Folds 1–6 (5,435 samples)
- Validation: Fold 7 (838 samples)
- Testing: Folds 8–10 (2,459 samples)

The dataset contains 10 environmental sound classes.

## 5. Training Commands

Run the training pipeline:

```bash
python -m src.train
```

This runs the implemented model training and comparison pipeline.

The four approaches are:

| Model | Input Features | Classifier |
|---|---|---|
| MFCC + SVM | MFCC features | SVM |
| YAMNet + SVM | YAMNet embeddings | SVM |
| YAMNet + MLP | YAMNet embeddings | MLP |
| CNN | Log-Mel Spectrograms | CNN |

Trained checkpoints are stored in the `checkpoints/` directory.

## 6. Evaluation Command

Run model evaluation:

```bash
python -m src.evaluate
```

Evaluation results and generated artifacts are stored in:

- `experiments/results/`
- `experiments/figures/`

The evaluation includes accuracy, Macro F1, Weighted F1, classification reports, and confusion matrix results.

### Random Seed Experiments

Three seed values were recorded: 42, 43, and 44.

The following table summarizes the current validation results:

| Model | Accuracy (Mean ± SD) | Macro F1 (Mean ± SD) |
|---|---:|---:|
| YAMNet + SVM | 81.03% ± 0.00% | 81.24% ± 0.00% |
| YAMNet + MLP | 77.76% ± 1.20% | 76.91% ± 1.75% |
| MFCC + SVM | 70.17% ± 0.00% | 69.45% ± 0.00% |
| CNN | 73.63% ± 2.21% | 73.95% ± 2.21% |

**Reproducibility note:**

- The SVM approaches produced identical results across the three recorded runs because they use deterministic training with the same data and hyperparameters.
- The MLP Seed 42 result came from the original hyperparameter-selection experiment, which evaluated three configurations. Seeds 43 and 44 used a fixed configuration. Therefore, the MLP mean and standard deviation are descriptive of the recorded runs and should not be interpreted as a fully controlled three-seed comparison.
- Seed 42 for the CNN also comes from the original training pipeline, while Seeds 43 and 44 were trained through the seed experiment script.

Detailed results are saved in:

- `experiments/results/seed_runs/seed_results.csv`
- `experiments/results/seed_runs/seed_summary.csv`

## 7. Demo Command

The project includes an interactive Gradio interface.

Run the demo:

```bash
python -m demo.app
```

Open the local URL displayed in the terminal, usually:

```text
http://127.0.0.1:7860
```

Upload an environmental audio recording to receive its predicted sound category.

The demo uses the selected YAMNet + SVM model.

**Demo limitation:** The current classifier predicts one of the 10 trained classes for every uploaded recording. It does not currently include an unknown or out-of-distribution sound rejection mechanism.

## 8. Hardware and Expected Runtime

### Hardware Used

| Component | Specification |
|---|---|
| Operating System | Windows 11 |
| CPU | Intel Core i7-13650HX |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB) |
| RAM | 16 GB |

TensorFlow GPU support was unavailable in the native Windows environment, so TensorFlow operations were executed on the CPU.

### Measured Inference Runtime

| Model | Measured Runtime | Measurement |
|---|---:|---|
| YAMNet + SVM | 2.48 ms/sample | 2,459 test samples |
| YAMNet + MLP | 0.042 ms/sample | 2,459 test samples |
| MFCC + SVM | 0.77 ms/sample | 2,459 test samples |
| CNN | 9.79 ms/sample | One batch of 32 samples |

**Runtime note:** These measurements are not directly comparable. The SVM measurements cover classifier inference on the test feature set, while the CNN measurement covers one batch. Audio preprocessing and feature extraction are not included in these classifier timings. Actual runtime depends on the hardware and input.

Training time was not consistently recorded for all final models.

## 9. Credits and References

### Dataset

Salamon, J., Jacoby, C., & Bello, J. P. (2014).  
A Dataset and Taxonomy for Urban Sound Research.  
Proceedings of the 22nd ACM International Conference on Multimedia.

DOI: https://doi.org/10.1145/2647868.2655045

UrbanSound8K dataset:
https://urbansounddataset.weebly.com/urbansound8k.html

### Pretrained Model

YAMNet, a pretrained audio event classification model provided by Google through TensorFlow Hub.

https://www.tensorflow.org/hub/tutorials/yamnet

### Frameworks and Libraries

- TensorFlow
- TensorFlow Hub
- Scikit-learn
- Librosa
- NumPy
- Pandas
- Matplotlib
- Gradio
- Joblib

All external datasets, pretrained models, frameworks, and libraries are acknowledged through their respective sources and documentation.