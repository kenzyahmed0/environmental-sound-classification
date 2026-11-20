
from pathlib import Path
import sys
import time
import random

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.datasets.loader import load_metadata
from src.datasets.splits import create_splits
from src.models.cnn import build_cnn_model, AudioSequence
from src.models.mfcc_svm import train_mfcc_svm


CACHE = ROOT / "experiments" / "cache"
OLD_RESULTS = ROOT / "experiments" / "results"
OUTPUT = ROOT / "experiments" / "results" / "seed_runs"

SEEDS = [42, 43, 44]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_cache(name):
    path = CACHE / f"{name}.npz"

    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")

    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def get_metrics(y_true, predictions):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true, predictions, average="macro"
        )
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Load the original Seed 42 validation results
    old = pd.read_csv(
        OLD_RESULTS / "validation_comparison.csv"
    )

    rows = []

    for _, item in old.iterrows():
        rows.append({
            "model": item["model"],
            "seed": 42,
            "accuracy": item["accuracy"],
            "macro_f1": item["macro_f1"],
            "source": "Original training result"
        })

    # Load cached features
    X_train_yamnet, y_train_yamnet = load_cache("train_yamnet")
    X_val_yamnet, y_val_yamnet = load_cache("validation_yamnet")

    X_train_mfcc, y_train_mfcc = load_cache("train_mfcc")
    X_val_mfcc, y_val_mfcc = load_cache("validation_mfcc")

    # Verify labels in the caches
    if not np.array_equal(y_train_yamnet, y_train_mfcc):
        raise ValueError("Training labels differ between caches.")

    if not np.array_equal(y_val_yamnet, y_val_mfcc):
        raise ValueError("Validation labels differ between caches.")

    # Load the same official train/validation split for CNN
    metadata = load_metadata()
    train_df, val_df, _ = create_splits(metadata)

    # Run Seeds 43 and 44
    for seed in [43, 44]:
        print(f"\n========== SEED {seed} ==========")
        set_seed(seed)

        # 1. YAMNet + SVM
        print("\nTraining YAMNet + SVM")

        start = time.time()

        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=10,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced"
            ))
        ])

        svm.fit(X_train_yamnet, y_train_yamnet)
        pred = svm.predict(X_val_yamnet)
        metrics = get_metrics(y_val_yamnet, pred)

        rows.append({
            "model": "YAMNet + SVM",
            "seed": seed,
            **metrics,
            "source": "Re-trained; deterministic model"
        })

        print(metrics, "Time:", round(time.time() - start, 2))

        # 2. YAMNet + MLP
        print("\nTraining YAMNet + MLP")

        start = time.time()

        mlp = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128),
                alpha=0.0001,
                learning_rate_init=0.001,
                activation="relu",
                solver="adam",
                batch_size=128,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=12,
                random_state=seed,
                verbose=False
            ))
        ])

        mlp.fit(X_train_yamnet, y_train_yamnet)
        pred = mlp.predict(X_val_yamnet)
        metrics = get_metrics(y_val_yamnet, pred)

        rows.append({
            "model": "YAMNet + MLP",
            "seed": seed,
            **metrics,
            "source": "Re-trained"
        })

        print(metrics, "Time:", round(time.time() - start, 2))

        # 3. MFCC + SVM
        print("\nTraining MFCC + SVM")

        start = time.time()

        mfcc_model = train_mfcc_svm(
            X_train_mfcc, y_train_mfcc
        )

        pred = mfcc_model.predict(X_val_mfcc)
        metrics = get_metrics(y_val_mfcc, pred)

        rows.append({
            "model": "MFCC + SVM",
            "seed": seed,
            **metrics,
            "source": "Re-trained; deterministic model"
        })

        print(metrics, "Time:", round(time.time() - start, 2))

        # 4. CNN from scratch
        print("\nTraining CNN")

        set_seed(seed)

        train_sequence = AudioSequence(
            train_df,
            batch_size=32,
            shuffle=True
        )

        val_sequence = AudioSequence(
            val_df,
            batch_size=32,
            shuffle=False
        )

        cnn = build_cnn_model()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]

        start = time.time()

        cnn.fit(
            train_sequence,
            validation_data=val_sequence,
            epochs=30,
            callbacks=callbacks,
            verbose=1
        )

        probabilities = cnn.predict(
            val_sequence,
            verbose=1
        )

        predictions = probabilities.argmax(axis=1)
        y_true = val_df["classID"].to_numpy()

        metrics = get_metrics(y_true, predictions)

        rows.append({
            "model": "CNN",
            "seed": seed,
            **metrics,
            "source": "Re-trained"
        })

        print(metrics, "Time:", round(time.time() - start, 2))

        del cnn
        tf.keras.backend.clear_session()

        # Save progress after each seed
        pd.DataFrame(rows).to_csv(
            OUTPUT / "seed_results.csv",
            index=False
        )

    # Final results
    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT / "seed_results.csv",
        index=False
    )

    summary = results.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std")
    ).reset_index()

    summary.to_csv(
        OUTPUT / "seed_summary.csv",
        index=False
    )

    print("\n========== SEED EXPERIMENTS COMPLETED ==========")
    print(results.to_string(index=False))
    print("\n========== SUMMARY ==========")
    print(summary.to_string(index=False))
    print("\nSaved results to:", OUTPUT)


if __name__ == "__main__":
    main()
