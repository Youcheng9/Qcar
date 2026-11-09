"""Build the early-fusion camera + LiDAR driving model.

The LiDAR points are rasterized into two bird's-eye-view channels and stacked
with the camera image before the shared CNN, so this is an input-level early
fusion model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
LIDAR_CHANNELS = 2  # occupancy and nearest normalized distance


def lidar_to_channels(points, mask, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Rasterize normalized car-frame LiDAR points into image-sized channels.

    ``points`` are the normalized values produced by SensorFusion/fusion.py:
    x is forward and y is left, both approximately in [-1, 1].  The result is
    not projected into the camera image; it is a fixed BEV representation that
    lets the early-fusion network consume both sensors without camera
    calibration.
    """
    points = np.asarray(points, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    batch = points.shape[0]
    out = np.zeros((batch, height, width, LIDAR_CHANNELS), dtype=np.float32)

    for b in range(batch):
        valid = mask[b] & np.isfinite(points[b]).all(axis=1)
        p = points[b, valid]
        if len(p) == 0:
            continue

        # x: forward [0, 1] -> image rows; y: left [-1, 1] -> image columns.
        x = np.clip(p[:, 0], 0.0, 1.0)
        y = np.clip(p[:, 1], -1.0, 1.0)
        row = np.clip(((1.0 - x) * (height - 1)).astype(np.int32), 0, height - 1)
        col = np.clip(((y + 1.0) * 0.5 * (width - 1)).astype(np.int32), 0, width - 1)

        # Multiple points can land in one cell; retain occupancy and nearest range.
        distance = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        for r, c, d in zip(row, col, distance):
            out[b, r, c, 0] = 1.0
            if out[b, r, c, 1] == 0.0 or d < out[b, r, c, 1]:
                out[b, r, c, 1] = d
    return out


def make_early_fusion_input(data):
    """Return the five-channel input expected by the early-fusion CNN."""
    image = data['image'].astype(np.float32) / 255.0
    lidar = lidar_to_channels(data['points'], data['mask'])
    return np.concatenate([image, lidar], axis=-1)


def build_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3 + LIDAR_CHANNELS)):
    """Create a model predicting steering, throttle, and stop from fused data."""
    inputs = keras.Input(shape=input_shape, name='camera_lidar')
    x = layers.Conv2D(24, 5, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(36, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(48, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    steering = layers.Dense(1, name='steering')(x)
    throttle = layers.Dense(1, name='throttle')(x)
    stop = layers.Dense(1, activation='sigmoid', name='stop')(x)

    model = keras.Model(inputs, [steering, throttle, stop])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'steering': keras.losses.Huber(),
            'throttle': keras.losses.Huber(),
            'stop': keras.losses.BinaryCrossentropy(),
        },
        loss_weights={'steering': 1.0, 'throttle': 0.5, 'stop': 1.0},
        metrics={'stop': [keras.metrics.BinaryAccuracy(name='accuracy')]},
    )
    return model
