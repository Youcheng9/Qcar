# Sensor Fusion: Camera + 2D LiDAR Dataset

We don't have a 3D LiDAR, so fusion here is geometric: the front camera image plus the RPLidar's 2D scan, turned into (x, y) points. There's no point-cloud work.

| File | Runs on | What it does |
|---|---|---|
| `fusion.py` | car + PC | Shared data format: polar to points, cleaning, normalize/pad, session writer/loader |
| `FusionCollectionServer.py` | QCar | Drive with the Logitech wheel and record synced camera, LiDAR and labels |
| `lidar_check.py` | QCar | Check the LiDAR orientation before collecting |
| `view_session.py` | PC (or car) | Session stats and a frame-by-frame viewer (camera + bird's-eye LiDAR) |
| `test_fusion.py` | PC | `python3 -m unittest test_fusion` |

## Data format

One folder per session, `data/<YYYYmmdd_HHMMSS>_<tag>/`:

```
labels.csv          index, t, image_t, lidar_t, image, lidar, steering, throttle, stop, tag
images/000123.jpg   front camera (CSI id 3), 420x220 BGR
lidar/000123.npz    angles (rad, raw RPLidar), distances (m), 384 readings
```

- `t` is when the sample was taken. `image_t` and `lidar_t` are when each sensor's data was captured, all from `time.time()` on the car. **skew** = `image_t - lidar_t`.
- `steering` and `throttle` are the values actually sent to the car. `stop` is 1 while the brake pedal is held.
- A sensor that fails leaves its field empty, and the row is still written, so failure examples are kept.
- The raw polar scan is stored, and points are computed at load time:
  `x = d*cos(θ)`, `y = d*sin(θ)` with `θ = FRONT - angle`. The car frame is x forward, y left.
  Readings that are `0`, `inf`, NaN, `< MIN_RANGE` or `> MAX_RANGE` are dropped. If `FRONT` turns out to be wrong, fix it in `fusion.py` and every session already recorded is corrected too.

Loading for training (numpy only, so it works with TF/Keras):

```python
import fusion, glob
d = fusion.load(sorted(glob.glob('data/*')), size=(160, 120))
d['image']     # (N,120,160,3) uint8
d['points']    # (N,384,2) float32 in [-1,1], zero-padded, sorted by bearing
d['mask']      # (N,384) bool, True = real point
d['steering'], d['throttle'], d['stop'], d['tag'], d['skew']
d['image_ok'], d['lidar_ok']   # False = missing/corrupt sensor (filled with zeros)
```

## Collection day checklist

1. `git pull` on the car, then `cd SensorFusion`.
2. **Orientation check.** Run `sudo python3 lidar_check.py`. Put a box about 0.5 m in front of the car: it should print `FRONT` with y ≈ 0. Move it to the car's left: it should print `LEFT`. If not, change `FRONT` in `fusion.py`.
3. **Drive.** Run `sudo python3 FusionCollectionServer.py --tag normal --control xbox` for the Logitech/Xbox wheel, or `sudo python3 FusionCollectionServer.py --tag normal --control wasd` for a keyboard connected to the QCar. You don't need to edit the server IP, `index_tracker.txt` or `images_catalogs.txt`.
   - Samples are recorded at 10 Hz (`--hz`) while the car is moving, or while the brake is held.
   - **Brake = stop label.** Hold the brake when the car should stop (person or obstacle ahead). The car stops, and the frames are labeled `stop=1`.
   - The XBOX button or Ctrl+C saves and quits in Xbox mode. In WASD terminal mode, use **W** to drive, **S** or **Space** to brake, **A/D** to steer, **R/F** to select reverse/forward, and **Q** to save and quit. Press **W** repeatedly to continue moving. Watch the console for `<-- check sensors`.
4. **Check right away.** Run `python3 view_session.py data/<session> --summary`. Look for: Hz ≈ 10, a median skew under ~0.1 s, almost no missing images or scans, and a mix of left and right steering.
5. Copy `data/` to a PC, for example `scp -r nvidia@<car-ip>:<path>/SensorFusion/data .`. If the folders are root-owned because of sudo, run `sudo chown -R nvidia data` first.

Record one session per scenario with `--tag`:

| tag | What to record |
|---|---|
| `normal` | Clean laps in both directions, plus some recoveries from the lane edge |
| `obstacle` | Boxes and cones on the track. Steer around them, or brake if the lane is blocked |
| `person` | Someone standing or walking in or near the lane. **Brake** (stop=1) when they're in the path, and keep recording while stopped |
| `lighting` | Lights off or dim, a flashlight or glare, shadows, a different time of day |
| `camera_fail` | Lens covered or partly covered while driving slowly |
| `lidar_fail` | LiDAR partly or fully blocked with cardboard. Don't unplug it: the server needs it at startup |

Get roughly balanced amounts of `stop=1` vs `stop=0` and of left vs right turns. Otherwise the model learns "always go straight".
