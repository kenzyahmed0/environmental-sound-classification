
from pathlib import Path
import time

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "experiments" / "cache"


def load_yamnet():
    """Load the pretrained YAMNet model."""
    model = hub.load(YAMNET_HANDLE)
    print("YAMNet loaded successfully.")
    return model


def audio_to_yamnet_embedding(path, yamnet_model):
    """Convert an audio file into one YAMNet embedding."""
    waveform, sample_rate = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    waveform = waveform.astype(np.float32)

    if waveform.size == 0:
        raise ValueError(f"Empty audio file: {path}")

    scores, embeddings, spectrogram = yamnet_model(
        tf.convert_to_tensor(waveform, dtype=tf.float32)
    )

    embedding = embeddings.numpy().mean(axis=0)

    return embedding.astype(np.float32)


def extract_or_load_embeddings(frame, cache_prefix, yamnet_model):
    """Extract embeddings or load them from cache if available."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_path = CACHE_DIR / f"{cache_prefix}_yamnet.npz"

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)

        paths = cached["paths"].astype(str)
        X = cached["X"].astype(np.float32)
        y = cached["y"].astype(int)

        current_paths = frame["filepath"].to_numpy(dtype=str)

        if np.array_equal(paths, current_paths):
            print("Loaded cached features:", cache_path)
            return X, y

        print("Cache mismatch. Rebuilding features.")

    features = []
    labels = []
    good_paths = []
    errors = []

    start_time = time.time()

    for i, row in enumerate(frame.itertuples(index=False), 1):
        try:
            embedding = audio_to_yamnet_embedding(
                row.filepath,
                yamnet_model
            )

            features.append(embedding)
            labels.append(int(row.classID))
            good_paths.append(row.filepath)

        except Exception as error:
            errors.append((row.filepath, repr(error)))

        if i % 250 == 0 or i == len(frame):
            print(
                f"{cache_prefix}: {i}/{len(frame)} processed | "
                f"Successful: {len(features)} | Errors: {len(errors)}"
            )

    if errors:
        error_df = pd.DataFrame(
            errors,
            columns=["filepath", "error"]
        )

        error_df.to_csv(
            CACHE_DIR / f"{cache_prefix}_errors.csv",
            index=False
        )

        raise RuntimeError(
            f"{len(errors)} audio files failed. "
            "Check the errors CSV file."
        )

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=int)

    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        paths=np.asarray(good_paths)
    )

    elapsed = time.time() - start_time

    print("Features shape:", X.shape)
    print(f"Saved features to: {cache_path}")
    print(f"Elapsed time: {elapsed:.1f} seconds")

    return X, y
