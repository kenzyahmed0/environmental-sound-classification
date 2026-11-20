
import time

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


SVM_CANDIDATES = [
    {"C": 1, "kernel": "rbf", "gamma": "scale"},
    {"C": 10, "kernel": "rbf", "gamma": "scale"},
    {"C": 30, "kernel": "rbf", "gamma": "scale"},
    {"C": 10, "kernel": "linear"},
]


def train_yamnet_svm(X_train, y_train, X_val, y_val):
    trials = []
    best_model = None
    best_score = -1
    best_params = None

    for params in SVM_CANDIDATES:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(**params, class_weight="balanced"))
        ])

        start = time.time()
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(y_val, predictions, average="macro")

        elapsed = time.time() - start

        trials.append({
            **params,
            "val_accuracy": accuracy,
            "val_macro_f1": macro_f1,
            "seconds": elapsed
        })

        print(
            params,
            "val_accuracy=", round(accuracy, 4),
            "val_macro_f1=", round(macro_f1, 4)
        )

        if macro_f1 > best_score:
            best_model = model
            best_score = macro_f1
            best_params = params

    results = pd.DataFrame(trials).sort_values(
        "val_macro_f1",
        ascending=False
    )

    print("Best YAMNet SVM parameters:", best_params)

    return best_model, results, best_params
