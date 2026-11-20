
from pathlib import Path
import json

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from src.datasets.loader import load_metadata
from src.datasets.splits import create_splits

from src.models.yamnet import (
    load_yamnet,
    extract_or_load_embeddings
)

from src.models.mfcc_svm import extract_mfcc_split

from src.models.cnn import AudioSequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read the selected model from validation
    config_path = RESULTS_DIR / "experiment_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            "Training configuration not found. Run src.train first."
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    winner_name = config["selected_model"]

    print("Model selected using validation:", winner_name)

    # 2. Load metadata and official test split
    print("\n========== Loading Test Dataset ==========")

    metadata = load_metadata()
    _, _, test_df = create_splits(metadata)

    y_test = test_df["classID"].to_numpy()

    print("Test samples:", len(test_df))

    # Class names ordered by class ID
    class_mapping = (
        metadata[["classID", "class"]]
        .drop_duplicates()
        .sort_values("classID")
    )

    class_ids = class_mapping["classID"].tolist()
    class_names = class_mapping["class"].tolist()

    # 3. Check and load all trained models
    print("\n========== Loading Trained Models ==========")

    model_paths = {
        "YAMNet + SVM": CHECKPOINTS_DIR / "yamnet_svm.joblib",
        "YAMNet + MLP": CHECKPOINTS_DIR / "yamnet_mlp.joblib",
        "MFCC + SVM": CHECKPOINTS_DIR / "mfcc_svm.joblib",
        "CNN": CHECKPOINTS_DIR / "cnn_model.keras"
    }

    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. Run training first."
            )

    models = {
        "YAMNet + SVM": joblib.load(model_paths["YAMNet + SVM"]),
        "YAMNet + MLP": joblib.load(model_paths["YAMNet + MLP"]),
        "MFCC + SVM": joblib.load(model_paths["MFCC + SVM"]),
        "CNN": tf.keras.models.load_model(model_paths["CNN"])
    }

    # 4. Extract YAMNet test embeddings
    print("\n========== Extracting YAMNet Test Features ==========")

    yamnet_model = load_yamnet()

    X_test_yamnet, y_test_yamnet = extract_or_load_embeddings(
        test_df,
        "test",
        yamnet_model
    )

    # 5. Extract MFCC test features
    print("\n========== Extracting MFCC Test Features ==========")

    X_test_mfcc, y_test_mfcc = extract_mfcc_split(
        test_df,
        "test"
    )

    # Check that extracted labels match the official test order
    if not (y_test_yamnet == y_test).all():
        raise ValueError("YAMNet test labels do not match test metadata.")

    if not (y_test_mfcc == y_test).all():
        raise ValueError("MFCC test labels do not match test metadata.")

    # 6. Evaluate the three feature-based models
    print("\n========== Evaluating YAMNet and MFCC Models ==========")

    test_inputs = {
        "YAMNet + SVM": (X_test_yamnet, y_test),
        "YAMNet + MLP": (X_test_yamnet, y_test),
        "MFCC + SVM": (X_test_mfcc, y_test)
    }

    test_rows = []
    predictions = {}

    for name in test_inputs:
        print(f"\nEvaluating {name}...")

        X_test, y_current = test_inputs[name]
        y_pred = models[name].predict(X_test)

        predictions[name] = y_pred

        test_rows.append({
            "model": name,
            "test_accuracy": accuracy_score(y_current, y_pred),
            "test_macro_f1": f1_score(
                y_current,
                y_pred,
                average="macro"
            ),
            "test_weighted_f1": f1_score(
                y_current,
                y_pred,
                average="weighted"
            )
        })

    # 7. Evaluate CNN batch by batch
    print("\n========== Evaluating CNN ==========")

    cnn_test_sequence = AudioSequence(
        test_df,
        batch_size=32,
        shuffle=False
    )

    cnn_probabilities = models["CNN"].predict(
        cnn_test_sequence,
        verbose=1
    )

    cnn_predictions = cnn_probabilities.argmax(axis=1)

    if len(cnn_predictions) != len(y_test):
        raise ValueError(
            "CNN predictions count does not match test samples."
        )

    predictions["CNN"] = cnn_predictions

    test_rows.append({
        "model": "CNN",
        "test_accuracy": accuracy_score(y_test, cnn_predictions),
        "test_macro_f1": f1_score(
            y_test,
            cnn_predictions,
            average="macro"
        ),
        "test_weighted_f1": f1_score(
            y_test,
            cnn_predictions,
            average="weighted"
        )
    })

    # 8. Save comparison of all four models
    test_results = pd.DataFrame(test_rows)

    test_results.to_csv(
        RESULTS_DIR / "test_results.csv",
        index=False
    )

    print("\n========== Test Results Comparison ==========")
    print(test_results.to_string(index=False))

    # 9. Report the model selected using validation
    if winner_name not in predictions:
        raise ValueError(
            f"Selected model '{winner_name}' was not evaluated."
        )

    print("\n========== Selected Model Report ==========")
    print("Selected model:", winner_name)

    selected_model_predictions = predictions[winner_name]

    report = classification_report(
        y_test,
        selected_model_predictions,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        RESULTS_DIR / "classification_report.csv"
    )

    print(
        classification_report(
            y_test,
            selected_model_predictions,
            labels=class_ids,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )

    # 10. Generate confusion matrix
    cm = confusion_matrix(
        y_test,
        selected_model_predictions,
        labels=class_ids
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_df.to_csv(
        RESULTS_DIR / "confusion_matrix.csv"
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False
    )

    ax.set_title(f"Confusion Matrix - {winner_name}")
    plt.tight_layout()

    figure_path = FIGURES_DIR / "confusion_matrix.png"

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # 11. Save evaluation summary
    summary = {
        "selected_model": winner_name,
        "selection_metric": config["selection_metric"],
        "test_samples": len(y_test),
        "test_accuracy": float(
            accuracy_score(y_test, selected_model_predictions)
        ),
        "test_macro_f1": float(
            f1_score(
                y_test,
                selected_model_predictions,
                average="macro"
            )
        ),
        "test_weighted_f1": float(
            f1_score(
                y_test,
                selected_model_predictions,
                average="weighted"
            )
        )
    }

    with open(
        RESULTS_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n========== Evaluation Completed ==========")
    print("Selected model:", winner_name)
    print("Results saved in:", RESULTS_DIR)
    print("Confusion matrix saved in:", figure_path)


if __name__ == "__main__":
    main()
