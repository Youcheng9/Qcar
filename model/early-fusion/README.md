# Early-fusion model

This folder contains the minimal training model for combining the QCar front
camera and 2D LiDAR data collected by `SensorFusion`.

The camera image is normalized and the LiDAR `(x, y)` points are rasterized
into occupancy and distance channels. Those five channels are stacked before
the CNN, which makes the model input-level early fusion. It does not project
LiDAR points into camera pixels, so camera calibration is not required.


Currently, the model is still a placeholder. Data has not been collected. Data needs to contain robust scenarios:
- lane recovery
- obstacle avoiding
- different lightings
- ccw and cw driving laps
- different obstacles

Run from the repository root:

```text
python3 model/early-fusion/train.py --data SensorFusion/data --output early_fusion.keras --tflite early_fusion.tflite
```

The model predicts steering, throttle, and stop from the human-labeled
training examples.

### Training pipeline

- **Validation split by session.** Whole sessions are held out per scenario
  tag (`--val-fraction`, default 0.2), so near-identical frames from one lap
  never appear in both train and validation. A tag needs at least 2 sessions
  to appear in validation; with no validation sessions, early stopping falls
  back to training loss.
- **Sample filtering.** Rows with a missing camera or LiDAR are dropped
  (`--keep-missing` keeps rows that still have one sensor, zero-filled), as
  are rows whose camera/LiDAR timestamps differ by more than `--max-skew`
  seconds (default 0.15).
- **On-the-fly batches.** Images stay uint8 in memory (~60 KB/sample) and the
  5-channel input is built per batch.
- **Augmentation.** Each training sample is mirrored left/right with
  probability 0.5 (image, LiDAR y, and steering sign flipped together;
  `--no-flip` disables). `--sensor-dropout p` blanks the camera or the LiDAR
  channels with probability p to train for sensor failure.
- **Deployment.** The QCar's Python 3.6 cannot load a `.keras` file, so use
  `--tflite` to export a TFLite model. `model.py`'s preprocessing
  (`fuse`, `lidar_to_channels`) needs only numpy, so the car builds the same
  input with `fuse(image[None], points[None], mask[None])`, where
  `points, mask = fusion.pad(fusion.to_points(angles, distances))`.

## Model Design

The model is a supervised early-fusion driving model. It learns from captured driving data and predicts the next action.


### Inputs

- Each training sample contains:
    - Camera: 120 × 160 × 3
    - LiDAR:  120 × 160 × 2
    - Total:  120 × 160 × 5

- The five channels are:
    1. Camera blue channel
    2. Camera green channel
    3. Camera red channel
    4. LiDAR occupancy
    5. Nearest LiDAR distance

The LiDAR points from SensorFusion/fusion.py are converted into a fixed bird’s-eye-view grid. Valid points are marked as occupied, and the nearest point’s normalized distance is stored in each grid cell.

The grid covers the 5 m × 10 m area in front of the car: forward is up and the car's left is on the left, the same left/right orientation as the camera. Points behind the car are dropped. Rows do not line up with camera rows, since the grid is a top-down view and not a projection into the image.

### Network

The combined five-channel input goes through shared convolutional layers:

```
5-channel input
      ↓
Conv2D: 24 filters
      ↓
Conv2D: 36 filters
      ↓
Conv2D: 48 filters
      ↓
Conv2D: 64 filters
      ↓
Global average pooling
      ↓
Dense layer: 64 units
      ↓
Three output heads
```

The camera and LiDAR are combined before these convolutional layers, which is why it is considered input-level early fusion.

### Outputs

The model predicts three values:

- steering  → continuous regression value
- throttle  → continuous regression value
- stop      → probability from 0 to 1
