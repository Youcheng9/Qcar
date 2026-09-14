from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


MODEL_HEIGHT = 67
MODEL_WIDTH = 420

BASE_DIR = Path(__file__).resolve().parent

WEIGHTS_PATH = (
    BASE_DIR
    / "Models"
    / "qcar_steering_weights.npz"
)

OUTPUT_PATH = (
    BASE_DIR
    / "Models"
    / "rightTurn.tflite"
)


def build_model():
    model = models.Sequential([
        layers.Input(
            shape=(MODEL_HEIGHT, MODEL_WIDTH, 1)
        ),

        layers.Conv2D(
            24,
            kernel_size=5,
            strides=2,
            activation="relu",
        ),

        layers.Conv2D(
            36,
            kernel_size=5,
            strides=2,
            activation="relu",
        ),

        layers.Conv2D(
            48,
            kernel_size=5,
            strides=2,
            activation="relu",
        ),

        layers.Conv2D(
            64,
            kernel_size=3,
            activation="relu",
        ),

        layers.Conv2D(
            64,
            kernel_size=3,
            activation="relu",
        ),

        layers.Flatten(),

        layers.Dense(
            100,
            activation="relu",
        ),

        layers.Dropout(0.2),

        layers.Dense(
            50,
            activation="relu",
        ),

        layers.Dense(
            10,
            activation="relu",
        ),

        layers.Dense(1),
    ])

    return model


def main():
    print("TensorFlow version:", tf.__version__)
    print("Weights:", WEIGHTS_PATH)
    print("Output:", OUTPUT_PATH)

    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            "Weights file not found: {}".format(
                WEIGHTS_PATH
            )
        )

    model = build_model()

    # Build the model variables before loading weights.
    model.predict(
        tf.zeros(
            (1, MODEL_HEIGHT, MODEL_WIDTH, 1),
            dtype=tf.float32,
        )
    )

    weight_file = np.load(
        str(WEIGHTS_PATH),
        allow_pickle=False,
    )

    weight_keys = sorted(
        weight_file.files,
        key=lambda name: int(name.split("_")[1]),
    )

    loaded_weights = [
        weight_file[key]
        for key in weight_keys
    ]   

    expected_weights = model.get_weights()

    print("Weight arrays in file:", len(loaded_weights))
    print("Weight arrays expected:", len(expected_weights))

    if len(loaded_weights) != len(expected_weights):
        raise ValueError(
            "Weight count mismatch: file contains {}, "
            "but model expects {}.".format(
                len(loaded_weights),
                len(expected_weights),
            )
        )

    for index, (loaded, expected) in enumerate(
        zip(loaded_weights, expected_weights)
    ):
        print(
            "Weight {}: loaded {} expected {}".format(
                index,
                loaded.shape,
                expected.shape,
            )
        )

        if loaded.shape != expected.shape:
            raise ValueError(
                "Shape mismatch at weight {}: "
                "loaded {}, expected {}".format(
                    index,
                    loaded.shape,
                    expected.shape,
                )
            )   

    model.set_weights(loaded_weights)

    print("Weights loaded successfully.")

    converter = (
        tf.lite.TFLiteConverter.from_keras_model(
            model
        )
    )

    # Avoid newer quantization/operator behavior initially.
    converter.optimizations = []

    tflite_model = converter.convert()

    with open(str(OUTPUT_PATH), "wb") as file:
        file.write(tflite_model)

    print("Converted model saved.")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
