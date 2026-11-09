"""Train the early-fusion driving model on SensorFusion session folders."""

import argparse
import glob
import os
import sys

import tensorflow as tf

# Allow this script to import the shared loader when run from this folder.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from SensorFusion import fusion
from model import make_early_fusion_input, build_model


def main():
    """Load sessions, train the model, and save a Keras model file."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Folder containing session folders')
    parser.add_argument('--output', default='early_fusion.keras')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    sessions = sorted(glob.glob(os.path.join(args.data, '*')))
    sessions = [p for p in sessions if os.path.isfile(os.path.join(p, 'labels.csv'))]
    if not sessions:
        raise SystemExit('No SensorFusion session folders found under ' + args.data)

    data = fusion.load(sessions, size=(160, 120))
    x = make_early_fusion_input(data)
    y = {
        'steering': data['steering'].astype('float32'),
        'throttle': data['throttle'].astype('float32'),
        'stop': data['stop'].astype('float32'),
    }

    model = build_model(input_shape=x.shape[1:])
    model.fit(
        x,
        y,
        validation_split=0.2,
        shuffle=True,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    model.save(args.output)
    print('Saved', args.output)


if __name__ == '__main__':
    main()
