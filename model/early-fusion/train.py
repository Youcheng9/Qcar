"""Train the early-fusion driving model on SensorFusion session folders."""

import argparse
import glob
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Allow this script to import the shared loader when run from this folder.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from SensorFusion import fusion
from model import fuse, build_model


def split_sessions(sessions, val_fraction, seed):
    """Hold out whole sessions, per scenario tag.

    Frames from one lap are nearly identical, so splitting by frame leaks them
    into validation. Splitting per tag keeps every scenario that has at least
    two sessions represented on both sides.
    """
    rng = random.Random(seed)
    by_tag = defaultdict(list)
    for s in sessions:
        rows = fusion.read_session(s)
        if rows:
            by_tag[rows[0]['tag']].append(s)

    train, val = [], []
    for tag in sorted(by_tag):
        group = sorted(by_tag[tag])
        rng.shuffle(group)
        k = int(round(len(group) * val_fraction))
        if len(group) >= 2:
            k = max(1, k)
        k = min(k, len(group) - 1)
        val += group[:k]
        train += group[k:]
        print('tag %-12s %d train / %d val sessions' % (tag, len(group) - k, k))
    return train, val


def filter_samples(data, max_skew, keep_missing):
    """Drop rows with a missing sensor (unless keep_missing) or a large camera/LiDAR time gap."""
    if keep_missing:
        keep = data['image_ok'] | data['lidar_ok']
    else:
        keep = data['image_ok'] & data['lidar_ok']
    keep &= ~(np.abs(data['skew']) > max_skew)  # NaN skew (a sensor missing) is not a skew failure
    print('kept %d / %d samples' % (keep.sum(), len(keep)))
    return {k: v[keep] for k, v in data.items()}


class FusionSequence(keras.utils.Sequence):
    """Builds fused 5-channel batches on the fly, so only uint8 images live in memory.

    With augment=True it shuffles each epoch, mirrors samples left/right
    (image, LiDAR y, steering sign) and optionally blanks one sensor to train
    for sensor failure.
    """

    def __init__(self, data, batch_size, augment=False, flip_prob=0.5,
                 sensor_dropout=0.0, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.data, self.batch_size, self.augment = data, batch_size, augment
        self.flip_prob, self.sensor_dropout = flip_prob, sensor_dropout
        self.rng = np.random.default_rng(seed)
        self.index = np.arange(len(data['steering']))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.index) / self.batch_size)

    def __getitem__(self, i):
        idx = self.index[i * self.batch_size:(i + 1) * self.batch_size]
        d = self.data
        image, points, mask = d['image'][idx], d['points'][idx], d['mask'][idx]
        steering = d['steering'][idx].astype('float32')

        if self.augment:
            flip = self.rng.random(len(idx)) < self.flip_prob
            image[flip] = image[flip][:, :, ::-1]
            points[flip, :, 1] *= -1.0
            steering[flip] *= -1.0

        x = fuse(image, points, mask)

        if self.augment and self.sensor_dropout > 0:
            drop = self.rng.random(len(idx)) < self.sensor_dropout
            camera = self.rng.random(len(idx)) < 0.5
            x[drop & camera, ..., :3] = 0.0
            x[drop & ~camera, ..., 3:] = 0.0

        y = {
            'steering': steering,
            'throttle': d['throttle'][idx].astype('float32'),
            'stop': d['stop'][idx].astype('float32'),
        }
        return x, y

    def on_epoch_end(self):
        if self.augment:
            self.rng.shuffle(self.index)


def export_tflite(model, path):
    """Write a TFLite copy for the QCar, which cannot run the .keras file."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(path, 'wb') as f:
        f.write(converter.convert())
    print('Saved', path)


def main():
    """Load sessions, train the model, and save a Keras model file."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Folder containing session folders')
    parser.add_argument('--output', default='early_fusion.keras')
    parser.add_argument('--tflite', default=None, help='Also export a .tflite model to this path')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-skew', type=float, default=0.15,
                        help='Drop samples whose camera/LiDAR timestamps differ by more (s)')
    parser.add_argument('--keep-missing', action='store_true',
                        help='Keep samples with one sensor missing (filled with zeros)')
    parser.add_argument('--sensor-dropout', type=float, default=0.0,
                        help='Probability of blanking one sensor per training sample')
    parser.add_argument('--no-flip', action='store_true', help='Disable left/right mirroring')
    args = parser.parse_args()

    sessions = sorted(glob.glob(os.path.join(args.data, '*')))
    sessions = [p for p in sessions if os.path.isfile(os.path.join(p, 'labels.csv'))]
    if not sessions:
        raise SystemExit('No SensorFusion session folders found under ' + args.data)

    train_sessions, val_sessions = split_sessions(sessions, args.val_fraction, args.seed)
    if not train_sessions:
        raise SystemExit('No non-empty sessions found under ' + args.data)

    size = (160, 120)
    train_data = filter_samples(fusion.load(train_sessions, size=size), args.max_skew, args.keep_missing)
    train = FusionSequence(train_data, args.batch_size, augment=True, seed=args.seed,
                           flip_prob=0.0 if args.no_flip else 0.5,
                           sensor_dropout=args.sensor_dropout)

    val = None
    if val_sessions:
        val_data = filter_samples(fusion.load(val_sessions, size=size), args.max_skew, args.keep_missing)
        if len(val_data['steering']):
            val = FusionSequence(val_data, args.batch_size)
    if val is None:
        print('WARNING: no validation sessions (each tag needs >= 2 sessions); '
              'early stopping uses training loss')

    model = build_model(input_shape=(size[1], size[0], 5))
    model.fit(
        train,
        validation_data=val,
        epochs=args.epochs,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor='val_loss' if val is not None else 'loss',
            patience=5, restore_best_weights=True)],
    )
    model.save(args.output)
    print('Saved', args.output)
    if args.tflite:
        export_tflite(model, args.tflite)


if __name__ == '__main__':
    main()
