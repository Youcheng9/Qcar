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
python3 model/early-fusion/train.py --data SensorFusion/data --output early_fusion.keras
```

The model predicts steering, throttle, and stop from the human-labeled
training examples.

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
