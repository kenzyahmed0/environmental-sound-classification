
from pathlib import Path
import json
import time

import joblib
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, f1_score

from src.datasets.loader import load_metadata
from src.datasets.splits import create_splits

from src.models.yamnet import (
    load_yamnet,
    extract_or_load_embeddings
)

from src.models.yamnet_svm import train_yamnet_svm
from src.models.yamnet_mlp import train_yamnet_mlp

from src.models.mfcc_svm import (
    extract_mfcc_split,
    train_mfcc_svm,
    evaluate_mfcc_svm
)

from src.models.cnn import (
    build_cnn_model,
    AudioSequence
)


# ==================================================
# Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
LOGS_DIR = PROJECT_ROOT / "experiments" / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


# ==================================================
# Helper function
# ==================================================

def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


# ==================================================
# Main training pipeline
# ==================================================

def main():

    start_time = time.time()

    # Set reproducibility seed
    tf.keras.utils.set_random_seed(42)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    print("\n========== Loading Dataset ==========")

    metadata = load_metadata()

    train_df, val_df, test_df = create_splits(metadata)

    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Test samples:", len(test_df))

    # --------------------------------------------------
    # 2. Load pretrained YAMNet
    # --------------------------------------------------

    print("\n========== Loading YAMNet ==========")

    yamnet_model = load_yamnet()

    # --------------------------------------------------
    # 3. Extract YAMNet embeddings
    # --------------------------------------------------

    print("\n========== Extracting YAMNet Features ==========")

    X_train_yamnet, y_train = extract_or_load_embeddings(
        train_df,
        "train",
        yamnet_model
    )

    X_val_yamnet, y_val = extract_or_load_embeddings(
        val_df,
        "validation",
        yamnet_model
    )

    # --------------------------------------------------
    # 4. Train YAMNet + SVM
    # --------------------------------------------------

    print("\n========== Training YAMNet SVM ==========")

    best_svm, svm_results, svm_params = train_yamnet_svm(
        X_train_yamnet,
        y_train,
        X_val_yamnet,
        y_val
    )

    svm_results.to_csv(
        RESULTS_DIR / "yamnet_svm_validation.csv",
        index=False
    )

    # --------------------------------------------------
    # 5. Train YAMNet + MLP
    # --------------------------------------------------

    print("\n========== Training YAMNet MLP ==========")

    best_mlp, mlp_results, mlp_params = train_yamnet_mlp(
        X_train_yamnet,
        y_train,
        X_val_yamnet,
        y_val
    )

    mlp_results.to_csv(
        RESULTS_DIR / "yamnet_mlp_validation.csv",
        index=False
    )

    # --------------------------------------------------
    # 6. Extract MFCC features
    # --------------------------------------------------

    print("\n========== Extracting MFCC Features ==========")

    X_train_mfcc, y_train_mfcc = extract_mfcc_split(
        train_df,
        "train"
    )

    X_val_mfcc, y_val_mfcc = extract_mfcc_split(
        val_df,
        "validation"
    )

    # --------------------------------------------------
    # 7. Train MFCC + SVM baseline
    # --------------------------------------------------

    print("\n========== Training MFCC SVM Baseline ==========")

    mfcc_model = train_mfcc_svm(
        X_train_mfcc,
        y_train_mfcc
    )

    mfcc_accuracy, mfcc_f1 = evaluate_mfcc_svm(
        mfcc_model,
        X_val_mfcc,
        y_val_mfcc
    )

    # --------------------------------------------------
    # 8. Train CNN from scratch
    # --------------------------------------------------

    print("\n========== Training CNN ==========")

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

    cnn_model = build_cnn_model()

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

    cnn_history = cnn_model.fit(
        train_sequence,
        validation_data=val_sequence,
        epochs=30,
        callbacks=callbacks,
        verbose=1
    )

    # CNN validation predictions
    cnn_probabilities = cnn_model.predict(
        val_sequence,
        verbose=1
    )

    cnn_predictions = cnn_probabilities.argmax(axis=1)

    cnn_true_labels = val_df["classID"].to_numpy()

    cnn_accuracy = accuracy_score(
        cnn_true_labels,
        cnn_predictions
    )

    cnn_f1 = f1_score(
        cnn_true_labels,
        cnn_predictions,
        average="macro"
    )

    print("CNN validation accuracy:", cnn_accuracy)
    print("CNN validation macro-F1:", cnn_f1)

    # Save CNN training history
    pd.DataFrame(cnn_history.history).to_csv(
        RESULTS_DIR / "cnn_training_history.csv",
        index=False
    )

    # Save CNN checkpoint
    cnn_model.save(
        CHECKPOINTS_DIR / "cnn_model.keras"
    )

    # --------------------------------------------------
    # 9. Compare models using validation Macro-F1
    # --------------------------------------------------

    print("\n========== Validation Comparison ==========")

    validation_results = pd.DataFrame([

        {
            "model": "YAMNet + SVM",
            "accuracy": float(
                svm_results.iloc[0]["val_accuracy"]
            ),
            "macro_f1": float(
                svm_results.iloc[0]["val_macro_f1"]
            )
        },

        {
            "model": "YAMNet + MLP",
            "accuracy": float(
                mlp_results.iloc[0]["val_accuracy"]
            ),
            "macro_f1": float(
                mlp_results.iloc[0]["val_macro_f1"]
            )
        },

        {
            "model": "MFCC + SVM",
            "accuracy": float(mfcc_accuracy),
            "macro_f1": float(mfcc_f1)
        },

        {
            "model": "CNN",
            "accuracy": float(cnn_accuracy),
            "macro_f1": float(cnn_f1)
        }

    ])

    validation_results = validation_results.sort_values(
        "macro_f1",
        ascending=False
    ).reset_index(drop=True)

    validation_results.to_csv(
        RESULTS_DIR / "validation_comparison.csv",
        index=False
    )

    print(validation_results.to_string(index=False))

    # Select winner using validation Macro-F1 only
    winner_name = validation_results.iloc[0]["model"]

    print("\nSelected model:", winner_name)

    # --------------------------------------------------
    # 10. Save all trained models
    # --------------------------------------------------

    print("\n========== Saving Models ==========")

    models = {
        "YAMNet + SVM": best_svm,
        "YAMNet + MLP": best_mlp,
        "MFCC + SVM": mfcc_model
    }

    # Save classical ML models
    for name, model in models.items():

        filename = (
            name.lower()
            .replace(" + ", "_")
            .replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            model,
            CHECKPOINTS_DIR / filename
        )

    # Save the selected model in its correct format
    if winner_name == "CNN":

        cnn_model.save(
            CHECKPOINTS_DIR / "best_model.keras"
        )

    else:

        winner_model = models[winner_name]

        joblib.dump(
            winner_model,
            CHECKPOINTS_DIR / "best_model.joblib"
        )

    # --------------------------------------------------
    # 11. Save experiment configuration
    # --------------------------------------------------

    config = {

        "dataset": "UrbanSound8K",

        "random_seed": 42,

        "split": {
            "train_folds": [1, 2, 3, 4, 5, 6],
            "validation_folds": [7],
            "test_folds": [8, 9, 10]
        },

        "models": {

            "yamnet_svm": svm_params,

            "yamnet_mlp": mlp_params,

            "mfcc_svm": {
                "C": 10,
                "kernel": "rbf",
                "gamma": "scale"
            },

            "cnn": {
                "architecture": "CNN",
                "input": "Log-Mel Spectrogram",
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "maximum_epochs": 30,
                "early_stopping_patience": 5,
                "dropout": 0.4
            }
        },

        "selected_model": winner_name,

        "selection_metric": "validation_macro_f1"
    }

    save_json(
        config,
        RESULTS_DIR / "experiment_config.json"
    )

    save_json(
        {
            "selected_model": winner_name,
            "selection_metric": "validation_macro_f1"
        },
        RESULTS_DIR / "winner.json"
    )

    # --------------------------------------------------
    # 12. Training summary
    # --------------------------------------------------

    elapsed = time.time() - start_time

    print("\n========== Training Completed ==========")

    print("Selected model:", winner_name)

    print(
        f"Total training time: {elapsed / 60:.2f} minutes"
    )

    print("Validation results:", RESULTS_DIR)
    print("Saved models:", CHECKPOINTS_DIR)


if __name__ == "__main__":
    main()
