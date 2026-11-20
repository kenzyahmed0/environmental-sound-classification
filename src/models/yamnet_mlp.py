
import time

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


SEED = 42

MLP_CANDIDATES = [
    {
        "hidden_layer_sizes": (256, 128),
        "alpha": 0.0001,
        "learning_rate_init": 0.001
    },
    {
        "hidden_layer_sizes": (512, 256),
        "alpha": 0.0005,
        "learning_rate_init": 0.001
    },
    {
        "hidden_layer_sizes": (256, 128),
        "alpha": 0.001,
        "learning_rate_init": 0.0005
    },
]


def train_yamnet_mlp(X_train, y_train, X_val, y_val):
    trials = []
    best_model = None
    best_score = -1
    best_params = None

    for params in MLP_CANDIDATES:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                **params,
                activation="relu",
                solver="adam",
                batch_size=128,
                max_iter=100,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=12,
                random_state=SEED,
                verbose=False
            ))
        ])

        start = time.time()
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(y_val, predictions, average="macro")

        elapsed = time.time() - start
        iterations = model.named_steps["clf"].n_iter_

        trials.append({
            **params,
            "val_accuracy": accuracy,
            "val_macro_f1": macro_f1,
            "iterations": iterations,
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

    print("Best YAMNet MLP parameters:", best_params)

    return best_model, results, best_params
