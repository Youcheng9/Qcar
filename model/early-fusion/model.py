"""Build the early-fusion camera + LiDAR driving model.

The LiDAR points are rasterized into two bird's-eye-view channels and stacked
with the camera image before the shared CNN, so this is an input-level early
fusion model.

The preprocessing functions (lidar_to_channels, fuse, make_early_fusion_input)
only need numpy, so the car can build model inputs without TensorFlow; only
build_model imports it.
"""

import numpy as np


IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120
LIDAR_CHANNELS = 2  # occupancy and nearest normalized distance


def lidar_to_channels(points, mask, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Rasterize normalized car-frame LiDAR points into image-sized channels.

    ``points`` are the normalized values produced by SensorFusion/fusion.py:
    x is forward and y is left, both approximately in [-1, 1].  The grid shows
    the forward half-plane seen from above: forward is up, and the car's left
    is on the left, matching the camera's left/right orientation.  Returns
    behind the car (x < 0) are dropped rather than clipped onto the bottom row.
    It is not a projection into camera pixels, so no calibration is needed.
    """
    points = np.asarray(points, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    batch = points.shape[0]
    out = np.zeros((batch, height, width, LIDAR_CHANNELS), dtype=np.float32)

    for b in range(batch):
        p = points[b]
        valid = (mask[b] & np.isfinite(p).all(axis=1)
                 & (p[:, 0] >= 0.0) & (p[:, 0] <= 1.0) & (np.abs(p[:, 1]) <= 1.0))
        p = p[valid]
        if len(p) == 0:
            continue

        # x: forward [0, 1] -> rows (far = top); y: left [1, -1] -> columns (left = 0).
        # Equal-width cells make negating y exactly mirror the columns, like a flipped image.
        row = np.minimum((1.0 - p[:, 0]) * height, height - 1).astype(np.int32)
        col = np.minimum((1.0 - p[:, 1]) * 0.5 * width, width - 1).astype(np.int32)

        # Multiple points can land in one cell; keep occupancy and nearest range.
        nearest = np.full((height, width), np.inf, dtype=np.float32)
        np.minimum.at(nearest, (row, col), np.hypot(p[:, 0], p[:, 1]))
        hit = np.isfinite(nearest)
        out[b, :, :, 0] = hit
        out[b, :, :, 1] = np.where(hit, nearest, 0.0)
    return out


def fuse(images, points, mask):
    """BGR uint8 images (N, H, W, 3) + padded points/mask -> (N, H, W, 5) float32."""
    image = np.asarray(images).astype(np.float32) / 255.0
    lidar = lidar_to_channels(points, mask, width=image.shape[2], height=image.shape[1])
    return np.concatenate([image, lidar], axis=-1)


def make_early_fusion_input(data):
    """Return the five-channel input expected by the early-fusion CNN."""
    return fuse(data['image'], data['points'], data['mask'])


def build_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3 + LIDAR_CHANNELS)):
    """Create a model predicting steering, throttle, and stop from fused data."""
    from tensorflow import keras
    from tensorflow.keras import layers

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
