# Dataset Documentation

## 1. Dataset Overview

This project uses the **UrbanSound8K** dataset for environmental sound classification.

UrbanSound8K is a publicly available dataset containing 8,732 labeled audio excerpts of urban environmental sounds. The recordings are organized into ten sound classes and were collected from field recordings uploaded to Freesound.

Each audio excerpt has a duration of four seconds or less. The dataset was designed to support research and benchmarking in environmental sound classification.

### Dataset Summary

| Attribute | Description |
|---|---|
| Dataset Name | UrbanSound8K |
| Version | 1.0 |
| Dataset Creators | Justin Salamon, Christopher Jacoby, and Juan Pablo Bello |
| Total Audio Clips | 8,732 |
| Number of Classes | 10 |
| Audio Format | WAV |
| Maximum Duration | 4 seconds |
| Metadata Format | CSV |
| Original Source | Freesound recordings |
| License | Creative Commons Attribution-NonCommercial 3.0 (CC BY-NC 3.0) |
| Dataset Role | Training, validation, and testing |

**Official dataset resources:**

- Dataset website: https://urbansounddataset.weebly.com/urbansound8k.html
- Original dataset description: https://serv.cusp.nyu.edu/projects/urbansounddataset/urbansound8k.html

## 2. Dataset Classes

The dataset contains ten environmental sound categories.

| Class ID | Class Name | Description |
|---:|---|---|
| 0 | Air Conditioner | Air conditioning and ventilation sounds |
| 1 | Car Horn | Vehicle horn sounds |
| 2 | Children Playing | Sounds of children playing |
| 3 | Dog Bark | Dog barking sounds |
| 4 | Drilling | Drilling and construction-related sounds |
| 5 | Engine Idling | Vehicle engine idling sounds |
| 6 | Gun Shot | Gunshot sounds |
| 7 | Jackhammer | Jackhammer and heavy construction sounds |
| 8 | Siren | Emergency and warning siren sounds |
| 9 | Street Music | Music recorded in urban environments |

The classification task is a ten-class supervised learning problem. Each audio recording is associated with one target class.

## 3. Data Source and Collection

UrbanSound8K was created by Justin Salamon, Christopher Jacoby, and Juan Pablo Bello.

The audio excerpts were obtained from field recordings uploaded to Freesound. The dataset creators selected and organized these excerpts into ten predefined folds.

The original recordings may differ in sampling rate, bit depth, and number of audio channels because these properties depend on the source recordings.

The dataset also provides metadata describing each audio excerpt, including its filename, source recording ID, start and end times, salience, fold number, and class label.

## 4. Dataset Structure and Metadata

The dataset contains two main components:

### 4.1 Audio Files

The audio recordings are provided in WAV format and organized into ten folders:

- fold1
- fold2
- fold3
- fold4
- fold5
- fold6
- fold7
- fold8
- fold9
- fold10

Each folder contains audio excerpts assigned to its corresponding fold.

### 4.2 Metadata File

The dataset includes a CSV file named:

`UrbanSound8K.csv`

The metadata contains the following fields:

| Field | Description |
|---|---|
| slice_file_name | Audio filename |
| fsID | Freesound recording identifier |
| start | Start time of the excerpt in the original recording |
| end | End time of the excerpt in the original recording |
| salience | Subjective indication of foreground or background sound |
| fold | Official fold number |
| classID | Numeric class identifier |
| class | Environmental sound class name |

The metadata file was used to identify audio files, retrieve their class labels, and construct the training, validation, and test splits.

## 5. Dataset License and Usage Conditions

UrbanSound8K is distributed under the **Creative Commons Attribution-NonCommercial 3.0 Unported (CC BY-NC 3.0)** license.

This license permits non-commercial use, subject to the applicable attribution requirements.

The original dataset documentation requests that academic research using UrbanSound8K acknowledge and cite the dataset publication.

The audio excerpts originate from Freesound recordings. Therefore, the original attribution information and applicable conditions associated with those recordings should also be respected.

The dataset is used in this project for academic and educational research purposes.

**License reference:**

https://creativecommons.org/licenses/by-nc/3.0/

## 6. Dataset Integrity and Verification

After downloading the dataset, its integrity was checked before preprocessing.

The following checks were performed:

| Verification | Result |
|---|---:|
| Expected number of metadata records | 8,732 |
| Actual metadata records | 8,732 |
| Number of sound classes | 10 |
| Existing audio files | 8,732 |
| Missing audio files | 0 |

The downloaded archive was also checked against its expected MD5 checksum:

`9aa69802bbf37fb986f71ec1483a196e`

The verification confirmed that the downloaded archive matched the expected checksum.

## 7. Data Preprocessing

The preprocessing pipeline prepares the original audio recordings and metadata for model training and evaluation.

The project uses model-specific feature extraction approaches because the four classification pipelines require different input representations.

### 7.1 Metadata Processing

The metadata CSV file was loaded and used to associate each audio filename with its corresponding class label and official fold.

The metadata was also used to organize the audio samples into the training, validation, and test sets.

### 7.2 Audio Feature Representations

The project investigates three audio representation approaches:

**1. YAMNet Embeddings**

A pretrained YAMNet model is used to extract audio embeddings. These embeddings are then used as input to the SVM and MLP classifiers.

**2. MFCC Features**

Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from audio recordings and used as input to the traditional SVM classifier.

**3. Log-Mel Spectrograms**

Log-Mel Spectrogram representations are used as input to the CNN trained from scratch.

These representations allow the project to compare pretrained audio features, traditional audio features, and learned spectrogram representations.

### 7.3 Feature Caching

Extracted YAMNet and MFCC features are stored in cache files to avoid repeating feature extraction during subsequent experiments.

The cached features are organized according to the corresponding dataset split:

- Training features
- Validation features
- Test features

This approach reduces repeated preprocessing operations and improves experiment efficiency.

## 8. Train, Validation, and Test Split Strategy

The project follows the official UrbanSound8K fold structure.

The dataset was divided as follows:

| Dataset Split | Official Folds | Number of Samples | Role |
|---|---|---:|---|
| Training | 1–6 | 5,435 | Model training |
| Validation | 7 | 838 | Hyperparameter tuning and model selection |
| Testing | 8–10 | 2,459 | Final model evaluation |
| **Total** | **1–10** | **8,732** | |

The predefined folds were preserved rather than randomly redistributing individual audio clips.

The validation set was used to compare model configurations and select the final model.

The test set was kept separate from model selection and used only for final performance evaluation.

This separation helps prevent information from the test set from influencing model selection.

Using the official folds also supports reproducibility and comparison with other UrbanSound8K experiments.

## 9. Dataset Distribution Considerations

UrbanSound8K contains an unequal number of recordings across its ten sound classes.

This class imbalance is important because a model may perform well on frequently represented classes while performing less effectively on classes with fewer examples.

For this reason, the project reports multiple evaluation metrics, including:

- Accuracy
- Macro F1-score
- Weighted F1-score

Macro F1-score is particularly useful because it gives equal importance to each class, regardless of the number of samples in that class.

The class-wise classification report and confusion matrix are also used to investigate model performance across different sound categories.

## 10. Reproducibility

The dataset download and preprocessing operations can be reproduced using the project scripts.

From the project root directory, run:

```bash
python -m data.download_dataset
```

Then execute:

```bash
python -m data.preprocess
```

The preprocessing process generates the metadata splits and processed dataset files used by the training pipeline.

The official fold assignments and dataset metadata are retained to ensure that experiments use the same data partitions.

## 11. Limitations of the Dataset

Several dataset characteristics may affect classification performance:

- **Class imbalance:** The number of audio samples differs across sound categories.
- **Recording variability:** Audio recordings have different sampling rates, bit depths, and channel configurations.
- **Background noise:** Some recordings contain overlapping or background sounds.
- **Short audio duration:** Each excerpt is limited to four seconds or less.
- **Acoustic similarity:** Different environmental sound classes may share similar acoustic characteristics.

These characteristics make UrbanSound8K a useful benchmark for evaluating environmental sound classification methods, while also introducing challenges for model generalization.

## 12. Dataset Citation

The following publication should be cited when using UrbanSound8K in academic research:

Salamon, J., Jacoby, C., & Bello, J. P. (2014). *A Dataset and Taxonomy for Urban Sound Research*. Proceedings of the 22nd ACM International Conference on Multimedia, 1041–1044.

https://doi.org/10.1145/2647868.2655045

## 13. Summary

UrbanSound8K provides 8,732 labeled audio excerpts across ten environmental sound categories.

In this project, the dataset was verified, organized using its official fold structure, and processed into feature representations suitable for different classification approaches.

The final data split contains 5,435 training samples, 838 validation samples, and 2,459 test samples.

The dataset supports the evaluation of pretrained audio embeddings, traditional MFCC features, and CNN-based spectrogram learning within a consistent environmental sound classification task.