
from pathlib import Path

import librosa
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "experiments" / "cache"


def mfcc_feature(path, n_mfcc=40):
    audio, sample_rate = librosa.load(
        path,
        sr=22050,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc
    )

    # Mean and standard deviation over time
    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1)
    ])

    return features.astype(np.float32)


def extract_mfcc_split(frame, name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = CACHE_DIR / f"{name}_mfcc.npz"
    current_paths = frame["filepath"].to_numpy(dtype=str)

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)

        paths = cached["paths"].astype(str)

        if np.array_equal(paths, current_paths):
            print("Loaded MFCC cache:", cache_path)
            return (
                cached["X"].astype(np.float32),
                cached["y"].astype(int)
            )

    features = []
    labels = []
    paths = []

    for i, row in enumerate(frame.itertuples(index=False), 1):
        feature = mfcc_feature(row.filepath)

        features.append(feature)
        labels.append(int(row.classID))
        paths.append(row.filepath)

        if i % 500 == 0 or i == len(frame):
            print(f"{name}: {i}/{len(frame)}")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=int)

    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        paths=np.asarray(paths)
    )

    print("Saved MFCC features:", cache_path)

    return X, y


def train_mfcc_svm(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=10,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    return model


def evaluate_mfcc_svm(model, X_val, y_val):
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(y_val, predictions, average="macro")

    print("MFCC+SVM validation accuracy:", accuracy)
    print("MFCC+SVM validation macro-F1:", macro_f1)

    return accuracy, macro_f1
