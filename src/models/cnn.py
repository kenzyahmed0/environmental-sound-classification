
import librosa
import numpy as np
import tensorflow as tf


SAMPLE_RATE = 22050
DURATION = 4.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

MAX_FRAMES = int(
    np.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH)
)


def audio_to_logmel(path):
    """
    Convert an audio file into a fixed-size Log-Mel spectrogram.
    """

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION
    )

    target_length = int(SAMPLE_RATE * DURATION)

    # Pad short audio or truncate long audio
    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )
    else:
        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Normalize each spectrogram
    mean = log_mel.mean()
    std = log_mel.std()

    log_mel = (log_mel - mean) / (std + 1e-6)

    # Ensure a fixed number of frames
    if log_mel.shape[1] < MAX_FRAMES:
        padding = MAX_FRAMES - log_mel.shape[1]
        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, padding))
        )
    else:
        log_mel = log_mel[:, :MAX_FRAMES]

    # Add the channel dimension
    return log_mel.astype(np.float32)[..., np.newaxis]


def build_cnn_model(
    input_shape=(N_MELS, MAX_FRAMES, 1),
    num_classes=10
):
    """
    Build a CNN for environmental sound classification.
    """

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(inputs)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="EchoMind_CNN"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


class AudioSequence(tf.keras.utils.Sequence):
    """
    Load audio spectrograms batch by batch.
    This avoids storing the entire dataset in RAM.
    """

    def __init__(
        self,
        frame,
        batch_size=32,
        shuffle=False
    ):
        super().__init__()

        self.paths = frame["filepath"].to_numpy()
        self.labels = frame["classID"].to_numpy().astype(np.int32)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(len(self.paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_paths = self.paths[batch_indices]
        batch_labels = self.labels[batch_indices]

        batch_features = np.stack([
            audio_to_logmel(path)
            for path in batch_paths
        ])

        return batch_features, batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

