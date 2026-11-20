
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

from src.models.yamnet import (
    load_yamnet,
    audio_to_yamnet_embedding
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "UrbanSound8K"
    / "metadata"
    / "UrbanSound8K.csv"
)

MODEL_PATH = CHECKPOINTS_DIR / "yamnet_svm.joblib"


def load_app_resources():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            "UrbanSound8K metadata not found."
        )

    metadata = pd.read_csv(METADATA_PATH)

    class_mapping = (
        metadata[["classID", "class"]]
        .drop_duplicates()
        .sort_values("classID")
    )

    id_to_class = dict(
        zip(
            class_mapping["classID"].astype(int),
            class_mapping["class"]
        )
    )

    model = joblib.load(MODEL_PATH)

    return model, id_to_class


MODEL, ID_TO_CLASS = load_app_resources()

YAMNET_MODEL = None


def predict_audio(audio_path):

    global YAMNET_MODEL

    if audio_path is None:
        return "Please upload an audio file."

    try:

        if YAMNET_MODEL is None:
            YAMNET_MODEL = load_yamnet()

        features = audio_to_yamnet_embedding(
            audio_path,
            YAMNET_MODEL
        )

        features = np.asarray(features).reshape(1, -1)

        predicted_id = int(MODEL.predict(features)[0])

        predicted_class = ID_TO_CLASS.get(
            predicted_id,
            "Unknown class"
        )

        return (
            f"Predicted sound: {predicted_class}\n"
            f"Class ID: {predicted_id}\n"
            f"Selected model: YAMNet + SVM"
        )

    except Exception as error:

        return f"Prediction failed: {str(error)}"


demo = gr.Interface(
    fn=predict_audio,
    inputs=gr.Audio(
        type="filepath",
        label="Upload an environmental sound"
    ),
    outputs=gr.Textbox(
        label="Prediction",
        lines=4
    ),
    title="EchoMind - Environmental Sound Classification",
    description=(
        "Upload an environmental sound recording "
        "to identify its sound category using YAMNet + SVM."
    )
)


if __name__ == "__main__":
    demo.launch()
