
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.loader import load_metadata
from src.datasets.splits import create_splits

CHECKPOINT = ROOT / "checkpoints" / "yamnet_svm.joblib"
CACHE = ROOT / "experiments" / "cache" / "test_yamnet.npz"
OUTPUT_DIR = ROOT / "experiments" / "results"
OUTPUT_FILE = OUTPUT_DIR / "misclassified_examples.csv"


def main():
    print("Loading test metadata...")
    metadata = load_metadata()
    _, _, test_df = create_splits(metadata)

    print("Loading cached YAMNet features...")
    cached = np.load(CACHE, allow_pickle=True)

    X_test = cached["X"]
    cached_paths = cached["paths"].astype(str)
    test_paths = test_df["filepath"].to_numpy(dtype=str)

    if not np.array_equal(cached_paths, test_paths):
        raise ValueError(
            "Test feature paths do not match test metadata order."
        )

    print("Loading selected model...")
    model = joblib.load(CHECKPOINT)

    print("Predicting test samples...")
    predictions = model.predict(X_test)

    results = test_df.copy()
    results["predicted_classID"] = predictions.astype(int)

    class_names = (
        metadata[["classID", "class"]]
        .drop_duplicates()
        .set_index("classID")["class"]
        .to_dict()
    )

    results["predicted_class"] = (
        results["predicted_classID"].map(class_names)
    )

    results["true_class"] = results["class"]
    results["correct"] = (
        results["classID"] == results["predicted_classID"]
    )

    errors = results[~results["correct"]].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    errors.to_csv(OUTPUT_FILE, index=False)

    print("\n========== ERROR ANALYSIS ==========")
    print("Total test samples:", len(results))
    print("Correct predictions:", int(results["correct"].sum()))
    print("Misclassified samples:", len(errors))

    print("\nMost common classification errors:")
    print(
        errors.groupby(
            ["true_class", "predicted_class"]
        ).size().sort_values(ascending=False).head(10)
    )

    print("\nFirst 15 misclassified examples:")
    print(
        errors[
            ["filepath", "true_class", "predicted_class"]
        ].head(15).to_string(index=False)
    )

    print("\nSaved misclassified examples to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
