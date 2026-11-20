
from pathlib import Path
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.loader import load_metadata
from src.datasets.splits import create_splits
from src.models.cnn import AudioSequence
from src.models.yamnet import load_yamnet, audio_to_yamnet_embedding


CHECKPOINTS = ROOT / "checkpoints"
CACHE = ROOT / "experiments" / "cache"
RESULTS = ROOT / "experiments" / "results"

OUTPUT_CSV = RESULTS / "computational_cost_measurements.csv"
OUTPUT_JSON = RESULTS / "computational_cost_details.json"

REPEATS = 5


def measure_sklearn_inference(model, X):
    # Warm-up
    model.predict(X[:min(10, len(X))])

    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        model.predict(X)
        times.append(time.perf_counter() - start)

    average_seconds = float(np.mean(times))

    return {
        "inference_total_seconds": average_seconds,
        "inference_ms_per_sample": (
            average_seconds / len(X) * 1000
        )
    }


def get_classifier(model):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("clf", model)

    return model


def get_parameter_info(name, model):
    classifier = get_classifier(model)

    if name == "CNN":
        total = int(model.count_params())
        trainable = int(
            sum(np.prod(weight.shape) for weight in model.trainable_weights)
        )

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "support_vectors": None
        }

    if hasattr(classifier, "coefs_"):
        total = sum(array.size for array in classifier.coefs_)
        total += sum(array.size for array in classifier.intercepts_)

        return {
            "total_parameters": int(total),
            "trainable_parameters": int(total),
            "support_vectors": None
        }

    if hasattr(classifier, "support_vectors_"):
        return {
            "total_parameters": None,
            "trainable_parameters": None,
            "support_vectors": int(len(classifier.support_vectors_))
        }

    return {
        "total_parameters": None,
        "trainable_parameters": None,
        "support_vectors": None
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Loading test metadata...")
    metadata = load_metadata()
    _, _, test_df = create_splits(metadata)

    print("Test samples:", len(test_df))

    print("\nLoading cached features...")
    yamnet_cache = np.load(
        CACHE / "test_yamnet.npz",
        allow_pickle=True
    )
    mfcc_cache = np.load(
        CACHE / "test_mfcc.npz",
        allow_pickle=True
    )

    X_yamnet = yamnet_cache["X"]
    X_mfcc = mfcc_cache["X"]

    if len(X_yamnet) != len(test_df) or len(X_mfcc) != len(test_df):
        raise ValueError("Cached feature count does not match test dataset.")

    print("Loading trained models...")
    models = {
        "YAMNet + SVM": joblib.load(CHECKPOINTS / "yamnet_svm.joblib"),
        "YAMNet + MLP": joblib.load(CHECKPOINTS / "yamnet_mlp.joblib"),
        "MFCC + SVM": joblib.load(CHECKPOINTS / "mfcc_svm.joblib"),
        "CNN": tf.keras.models.load_model(CHECKPOINTS / "cnn_model.keras")
    }

    inputs = {
        "YAMNet + SVM": X_yamnet,
        "YAMNet + MLP": X_yamnet,
        "MFCC + SVM": X_mfcc
    }

    rows = []

    print("\nMeasuring model inference and parameter counts...")

    for name, model in models.items():
        print("Measuring:", name)

        if name == "CNN":
            cnn_sequence = AudioSequence(
                test_df.iloc[:32].reset_index(drop=True),
                batch_size=32,
                shuffle=False
            )

            X_cnn, _ = cnn_sequence[0]

            # Warm-up
            model.predict(X_cnn, verbose=0)

            times = []

            for _ in range(REPEATS):
                start = time.perf_counter()
                model.predict(X_cnn, verbose=0)
                times.append(time.perf_counter() - start)

            average_seconds = float(np.mean(times))

            inference = {
                "inference_total_seconds": average_seconds,
                "inference_ms_per_sample": (
                    average_seconds / len(X_cnn) * 1000
                )
            }

        else:
            inference = measure_sklearn_inference(
                model,
                inputs[name]
            )

        parameter_info = get_parameter_info(name, model)

        checkpoint_path = {
            "YAMNet + SVM": CHECKPOINTS / "yamnet_svm.joblib",
            "YAMNet + MLP": CHECKPOINTS / "yamnet_mlp.joblib",
            "MFCC + SVM": CHECKPOINTS / "mfcc_svm.joblib",
            "CNN": CHECKPOINTS / "cnn_model.keras"
        }[name]

        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

        rows.append({
            "model": name,
            **inference,
            **parameter_info,
            "checkpoint_size_mb": round(size_mb, 3)
        })

    print("\nMeasuring YAMNet feature extraction separately...")

    yamnet_model = load_yamnet()

    sample_paths = test_df["filepath"].iloc[:20].tolist()

    start = time.perf_counter()

    for path in sample_paths:
        audio_to_yamnet_embedding(path, yamnet_model)

    yamnet_extraction_seconds = time.perf_counter() - start

    extraction_info = {
        "audio_samples": len(sample_paths),
        "total_seconds": yamnet_extraction_seconds,
        "average_seconds_per_audio": (
            yamnet_extraction_seconds / len(sample_paths)
        )
    }

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False)

    details = {
        "hardware": {
            "cpu": "Intel Core i7-13650HX",
            "ram_gb": 16,
            "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU 6GB",
            "operating_system": "Windows 11"
        },
        "measurement_method": {
            "inference_repeats": REPEATS,
            "classifier_inference_samples": len(test_df),
            "cnn_inference_batch_size": 32,
            "cnn_inference_samples": 32,
            "yamnet_feature_extraction_samples": 20
        },
        "yamnet_feature_extraction": extraction_info,
        "models": rows
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as file:
        json.dump(details, file, indent=4)

    print("\n========== COMPUTATIONAL COST RESULTS ==========")
    print(results_df.to_string(index=False))

    print("\nYAMNet feature extraction:")
    print(extraction_info)

    print("\nSaved CSV:", OUTPUT_CSV)
    print("Saved JSON:", OUTPUT_JSON)


if __name__ == "__main__":
    main()
