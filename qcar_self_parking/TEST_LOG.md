# QCar 1 Self-Parking – Test Log

## 2026-11-02

**Focus:** gyro calibration, stand rehearsals of vehicles mode, first floor run, LiDAR debugging.
**Mode tested:** `--mode vehicles --entry reverse --side right` (the defaults: park in a gap between boxes on the right, backing in).

### Summary
- **Gyro is working correctly.** A dedicated test (`gyro_check.py`) showed:
  - it reports in rad/s, as the code assumes
  - its sign is correct: positive when the car turns left
  - it is simply a noisy, coarse sensor (about ±0.09 rad/s of jitter at rest)

  The original calibration limit was too strict for this sensor. After loosening it and averaging for longer, calibration passes on the first try, and the heading drift while driving straight fell from about 0.5°/s to about 0.04–0.2°/s.
- **The parking sequence runs end to end on the stand.** With the wheels off the ground and a box held beside the car as a "parked car", the car:
  1. detected the row
  2. found a gap
  3. made a plan
  4. drove to the turn point
  5. ran the reversing turn
  6. entered the spot and finished with the hazards on

  The safety stop reliably blocked reversing toward an obstacle. Some results only look wrong on a stand: the turn "never reaching 90°" and a −94° final heading error, because the body can't rotate there.
- **The first floor run hesitated constantly.** Saving the output to `floor1.log` showed the safety stop had blocked the car in about 70% of control loops, both driving forward and reversing, in open space. As a result:
  - the search crawled at a fifth of its intended speed
  - the turn took about three times as long as it should
  - the car stopped right after the turn instead of backing into the spot
- **Root cause of the hesitation: made-up LiDAR readings.** Quanser's `QCarLidar` class filters every scan by default. On the car's older numpy, that filter silently fails to remove invalid readings and then blends them into the real ones. The result is short distances (a few cm to about 1 m) appearing randomly in every direction. Our code now turns that filter off and discards invalid readings itself. This is verified in simulation but **not yet on the car**.
- **First real-world estimate of the turn radius: about 0.42 m.** During the floor run's 90° turn, the car turned 84° over 0.62 m of wheel travel. The parking plan currently assumes 0.47 m (calculated from the wheelbase, never measured), so spots would come out slightly off-centre. This is only reliable once the odometer scale is confirmed.

---

### 1. Gyro check (`gyro_check.py`)
**Procedure:**
- **Phase 1 (5 s):** car left completely still on a solid surface.
- **Phase 2:** car turned by hand exactly 90° to the left, then held still, while the script added up the gyro's z-axis readings.

| Measurement | Result | What it means |
|---|---|---|
| Sample rate | 412 samples in 5 s (12.1 ms between reads), 0% repeated | The sensor delivers fresh data on every read |
| z at rest | mean −0.0167, std **0.0924** rad/s | A small constant offset (bias) plus a lot of jitter |
| z resolution | values move in steps of about 0.017 rad/s (about 1°/s) | Coarse sensor: the jitter is a few steps up and down |
| 90° left turn | integrated z = **+1.475** (90° = 1.571 rad) | Units are **rad/s**, as the code assumes; no conversion needed |
| Sign | positive for a left turn | `GYRO_Z_SIGN = +1` in config.py is correct |

**Conclusion:**
- The gyro calibration had failed 3 times in a row with "too noisy". That wasn't because the car was moving: the noise limit (`GYRO_CAL_MAX_STD = 0.01`) was smaller than a single step of the sensor, so no amount of holding still could pass it.
- The noise mostly averages out over time. After the hand turn, the integrated angle drifted only from 1.549 to 1.475 over about 12 s of holding still.

### 2. Gyro calibration tuning (stand runs)
**Background:** at the start of every run the car holds still and averages the gyro to find its bias, which is then subtracted from every reading. A poor bias estimate makes the heading slowly drift, which would make the car curve during the search and stop its 90° turn at the wrong angle.

| Calibration time | Bias [rad/s] | Heading drift over ~14 s of searching (car not turning) |
|---|---|---|
| 5 s | −0.01073 | +7.4°, a steady climb (~0.5°/s) |
| 15 s | −0.01825 | wandered −1.2…+2.0°, ended at +0.5° (~0.04°/s) |
| 15 s | −0.00202 | +3.3° (~0.2°/s) |
| 15 s | −0.00817 | stayed within ±1.1° |

**Conclusion:**
- The steady drift at 5 s came from too few samples to average out the gyro's jitter, not from vibration or electrical noise from the running motor.
- At 15 s the drift is small and no longer builds up steadily. Over a full parking run that's a heading error of a few degrees at most.
- The remaining run-to-run variation (0 to about 0.3°/s) is normal for this sensor.
- **Cost:** each run now waits 15 s at the start, during which nobody should touch the car.

### 3. Stand rehearsals (car on a stand, box held beside it)
- **Control loop timing:** 10.1–10.3 ms on average, so the loop keeps its full 100 Hz rate.
  - Reading the LiDAR returns immediately instead of waiting for a new scan.
  - The occasional maximum of 20–25 ms only happens in the first second.
- **Speed control:** the odometer rose by almost exactly 0.20 m per second during the search, matching the target search speed. This proves the speed controller follows the motor's speed reading (`motorTach`). It does **not** yet prove that reading is in real metres per second (open item 4).
- **Row detection:** the right-hand LiDAR beam picked up the "row" at 0.38 m (an unplanned object near the stand), then 0.96 m and 0.73 m (the box).
  - **0.38 m would have been rejected** for parking: a reverse-in needs the row at least 0.60 m from the LiDAR, so the car can finish its turn before reaching the neighbouring cars.
- **Gaps that weren't detected:**
  - One run: no gap at all after the row was seen.
  - Next run: a gap that opened and closed again after 0.01 m.

  In both, after the box was pulled away, something else was still within about 0.2 m of the box's distance: most likely the person holding it, or their hand, still in the side beam. A gap only counts when the side reading jumps more than 0.20 m beyond the row or shows nothing. To make this easier to diagnose, the `[TIMING]` log lines now show the live side distance and the gap threshold.
- **Full sequence reached:** POSITION → TURN → ENTER → DONE.
  - **TURN** ended with `arc length exceeded, heading only −3.8 deg`. That's the backup rule working: on a stand the car can't rotate, so it stops turning after 1.3× the expected arc length of wheel travel.
  - **DONE** showed a heading error of −94.1° for the same reason.
  - **During TURN** there were dozens of `[SAFETY] obstacle in path during TURN - holding` messages, so the car reversed in short bursts.
  - **ENTER** ended immediately with "reached end of spot".
  - At the time, both were blamed on a person standing behind the car. Section 5 shows the real cause was most likely the bad LiDAR data.
- **Setup rules learned for rehearsals and floor tests:**
  - **Row distance:** 0.6–1.0 m from the LiDAR (centre of the car) to the box fronts. Under 0.60 m the car refuses to park; over 1.5 m it doesn't see a row.
  - **Gap width:** 0.45–0.55 m.
    - Under 0.39 m it's rejected as too narrow.
    - At about 0.58 m or more it's treated as open space after the last box.
  - **A box after the gap,** or the car never knows the gap has ended.
  - **Box size:** the boxes must be taller than the LiDAR and at least as deep as the car.
  - **Clear behind:** keep 0.5 m clear behind the car while it reverses, and stand on the car's left (away from the row), not behind it.
- **Logging workflow:** the terminal only keeps the last 1000 lines by default, and the safety messages pushed the start of the run out of view. All runs are now saved with:
  ```
  sudo python3 -u qcar_self_parking.py --mode vehicles 2>&1 | tee floorN.log
  ```
  Then copy the log to the PC with `scp` for analysis.

### 4. First floor run (`floor1.log`)
| Item | Value | Notes |
|---|---|---|
| Safety-stop lines | 2,992 (SEARCH 1,192 · POSITION 771 · TURN 1,029) | Blocked in both directions, in open space: the first sign of bad LiDAR data |
| Search speed | ~0.04 m/s instead of 0.20 | The car moved only in the loops where it wasn't blocked |
| Time to reach the turn point | ~11 s for 0.24 m | Should take ~2 s |
| Gap | stayed open for ~0.58 m | Treated as open space after the last box; parked in a 0.44 m spot beside that box |
| Plan | row 0.74 m, spot centre at x = 0.57 m, turn start at x = 1.03 m, enter 0.64 m after the arc | The planning maths ran as designed |
| Turn | heading 4.1° → 88.1° over 0.62 m of wheel travel (~19 s, should be ~6 s) | **Turn radius ≈ 0.42 m**, if the odometer scale is right |
| Enter | stopped on the first loop with "reached end of spot" | A false close reading behind the car looked like the back of the spot; 0.64 m of reversing never happened |
| Final | heading error −2.0°, left 0.89 m, right 0.69 m | Heading was good. The car ended outside the spot (no box close on either side) |

**Positive results:**
- **The gyro tracks a real turn correctly while driving:** `heading` rose smoothly to 88° as the car turned.
- **The turn itself ended on heading** (88.1° against a 90° target), not by the arc-length backup.

### 5. LiDAR root cause
- **What `calibrate_lidar.py` showed on the floor:** this script was extended to show the closest reading in each safety sector and whether it would stop the car. With the car in open space:
  - every direction (front, rear, left, right) jumped between about 0.02 m and 2 m from one scan to the next, every 0.1 s
  - `STOP` appeared on most lines

  Readings under about 0.15 m are physically impossible for an RPLidar, so this was clearly invalid data, not real obstacles.
- **Where it comes from:**
  - Quanser's `QCarLidar` class (`pal/products/qcar.py`) has a filter, `filter_rplidar_data()`, that is turned on by default.
  - The filter is meant to remove invalid readings (reported as distance 0) with `np.delete(angles, distances == 0)`, then interpolate the scan onto an even 1° grid.
  - On the car's older numpy, `np.delete` treats the True/False list as the numbers 1 and 0. So it deletes only the first two readings and keeps all the zeros. The `FutureWarning` printed at `qcar.py:531` on every run was warning about exactly this.
  - The interpolation then blends real distances with those zeros, producing random short readings everywhere.
- **Why it caused the symptoms:**
  - The safety check uses the **closest** reading in a ±15° sector, so a single made-up short value stops the car. That explains the constant holds and the ENTER problem.
  - Gap detection uses the **median** of the side readings, which ignores occasional bad values. That's why row and gap detection mostly worked.
- **The fix:** our LiDAR wrapper now creates the LiDAR with `QCarLidar(..., enableFiltering=False)`, getting the raw scan. That's the same data the Qcar-master project used successfully through its own `lidar2.py`. Our code then discards readings under 5 cm itself, which removes the zeros. The broken filter only resamples the scan and doesn't rotate or flip angles, so the direction settings (`FRONT_ANGLE_DEG = 90`, `ANGLE_DIR = -1`) should still be valid.
- **Verification (simulation):** a simulated room with walls 1.0–2.0 m away, with one third of the readings invalid, over 200 scans:
  - **New code:** front/rear exactly 1.00 m and left/right 1.97–1.98 m on every scan.
  - **Quanser's filter, with old-numpy behaviour:** 0.05–1.3 m at random, matching what the real car showed.

---

### Code and config changes
| File | Change | Why |
|---|---|---|
| `parking/config.py` | `GYRO_CAL_MAX_STD` 0.01 → **0.15** rad/s | The sensor's resting noise is ~0.09 rad/s; 0.15 still catches real bumps (turning reads >1 rad/s) |
| `parking/config.py` | `GYRO_CAL_DURATION` 2 → 5 → **15** s | Longer averaging gives a better bias estimate and less heading drift (section 2) |
| `parking/config.py` | new `SAFETY_LOG_PERIOD = 0.5` s | At most 2 `[SAFETY]` messages per second instead of one every loop (up to 100/s) |
| `parking/config.py` | new `LIDAR_MIN_VALID = 0.05` m | Drops invalid zero readings and anything closer than the LiDAR's own housing |
| `parking/lidar.py` | `enableFiltering=False`; new `nearest()` method (distance and angle of the closest reading) | Avoids the broken Quanser filter (section 5); `nearest()` powers the new diagnostics |
| `parking/parker.py` | `[TIMING]` lines show the live side distance, row distance and gap threshold | Shows why a gap did or didn't open during rehearsals |
| `parking/parker.py` | `[SAFETY]` / `[ENTER]` messages say which sensor (front/rear), the distance, angle and limit, and how many loops were blocked | Shows exactly what triggered a stop and how often |
| `parking/calibration.py` | `calibrate_lidar.py` also shows each safety sector's closest reading with `STOP`/`ok`, and the closest reading all around | Finds false close readings or parts of the car being detected |
| `gyro_check.py` (new) | Measures gyro noise per axis, units (rad/s vs deg/s) and z sign | Section 1 |

---

### Open items / next session
1. **Copy the `parking/` folder to the car.** The LiDAR fix, the new log messages and the 15 s gyro setting in the PC copy are only on the PC so far:
   ```
   scp -r "C:\Users\rafae\OneDrive\Documents\qcar_self_parking\parking" nvidia@<CAR_IP>:~/Documents/qcar_self_parking/
   ```
2. **Confirm the LiDAR fix with `calibrate_lidar.py` in open space.** The numbers should be steady, with no `STOP` while nothing is near the car. Then redo the Stage 1 LiDAR checks:
   - a book at the front, left, rear and right; only the matching direction should drop
   - a tape-measured distance to a wall should match the reading within a few cm
   - with the car clear, front > 0.29 m and rear > 0.30 m
3. **Repeat the floor run with `tee`.**
   - **Expect:** no hesitation, the search at full speed, and ENTER backing the full planned distance into the spot.
   - **Record:** the actual gap width and box positions, so the plan can be checked against reality.
4. **Check the odometer scale.** Put a tape mark on a driven wheel and count its turns over 10 s of stand running. Turns × circumference (π × wheel diameter) should equal the reported distance (about 2.0 m). If it doesn't, every distance in the plan is off by that factor.
5. **Measure the turn radius.** Drive at full steering lock, mark the circle on the floor, and measure it. Compare with the ~0.42 m estimate, then update `TURN_RADIUS` in config.py (currently 0.47 m) so the car lines up with the centre of the spot.
6. **Remaining Stage 2 checks:**
   - **LEDs:** right turn signal during the search, brake lights when stopping, hazards at the end.
   - **Heading hold:** turn the car left during the search; `heading` should rise and the front wheels should steer right to correct.
   - **Stopping mid-run:** Ctrl+C while moving should stop the car immediately.
7. **Camera work (empty-lot mode only; not needed for vehicles mode):**
   - **Camera ID:** confirm ID 0 is really the right-side camera with the raw-frame command.
   - **Lens:** calibrate with the on-screen checkerboard (`calibrate_intrinsics.py`).
   - **Floor view:** set up the bird's-eye view with floor markers (`make_bev.py`).
   - **Mount position:** measure `CAMERA_MOUNT`.
   - **Detection:** tune it with `calibrate_camera.py`.

---

## 2026-11-05

**Focus:**
- Morning: confirmed the LiDAR fix on the floor, in vehicles mode.
- Rest of the day: restarted the camera-based (empty-lot) mode: camera identification and lens calibration, plus tools for the bird's-eye floor setup.

**Modes:** `--mode vehicles --entry reverse --side right` (driving); camera work was done with the car stationary.

### Summary
- **The LiDAR fix works on the real car, and vehicles mode now parks.** ("LiDAR fix" = the change from 2026-11-02, section 5: Quanser's `QCarLidar` scan filter is turned off with `enableFiltering=False`, because on the car's older numpy it left invalid zero readings in and blended them into fake short distances; our own code now drops readings under 5 cm instead. That had made the safety stop trigger constantly.)
  - The run log (`floor2.log`) has no `[SAFETY]` stops at all; the first floor run had 2,992. The search ran at its full 0.20 m/s, and ENTER backed in continuously until it reached a real obstacle at the back of the spot.
  - The car finished **within 1° of straight** (heading error −0.9°), between the two neighbouring obstacles, about 2–3 cm off-centre.
- **Camera IDs confirmed.** A new script (`camera_snapshot.py`) saved one picture from each of the four cameras: ID 0 = right, 1 = rear, 2 = left, 3 = front. That matches `CAMERA_ID` in config.py, so nothing needed changing.
- **Right-camera lens calibrated (good result on the second attempt).**
  - **First attempt:** looked fine (RMS 0.30 px), but the board only covered 56% of the image. The fitted lens model broke down before the edges of the image, including the bottom corners where the floor is.
  - **Fix:** `calibrate_intrinsics.py` now detects this problem and offers a more robust `--simple` lens model.
  - **Second attempt:** 40 frames with `--simple`, RMS 0.43 px, and a model that is valid all the way to every edge and corner.
- **Found a camera-view problem early.** A cable was looping in front of the right camera, blocking a large part of its view. It was moved out of the way before calibrating.
- **Built a tool for the bird's-eye floor step.** `bev_point_picker.html` is a click-to-pick page that writes the floor-marker points file, so marker pixel positions no longer have to be read by hand. A grid layout for the floor markers also means their positions are known without measuring each one.

---

### 1. Vehicles mode: floor run 2 (`floor2.log`)
| Step | What happened | Notes |
|---|---|---|
| Gyro calibration | Passed first try: bias −0.0184 rad/s, std 0.108 | The 15 s calibration is still good |
| Search | Row seen at 1.34 m, settling to 1.45 m; gap opened at x = 0.62 m with the side reading at ~1.9 m | **Row was near the 1.5 m detection limit**; recommended range is 0.6–1.0 m |
| Gap | Still open after ~0.58 m, so it was treated as open space after the last box, using a 0.44 m spot right beside it | The box after the gap wasn't seen before the 0.58 m cut-off: either the gap was wider than ~0.58 m or that box was too far away |
| Plan | Spot centre x = 0.84 m, turn start x = 1.31 m, **enter 1.36 m** after the arc | The long entry distance comes from the far row |
| Position | Reached the turn point in ~2 s, heading 0.4° | Normal speed: no safety holds |
| Turn | Ended at **96.4°** (target 90°) | ~6° overshoot: the turn stops at 88°, but the car keeps rolling at full lock while it brakes |
| Enter | Heading hold brought the heading from 96.4° back to ~89–90°; stopped when a real obstacle was 0.29 m behind the LiDAR (limit 0.30 m) | Stopped by the back of the spot before the planned 1.36 m |
| Done | Heading error **−0.9°**, left 0.15 m, right 0.20 m | Parked straight, slightly closer to the left neighbour |

**Question raised:** odometer and heading disagreed during part of the turn.
- **First second:** 0.127 m of travel and 18° of turn, consistent with a ~0.42 m turn radius.
- **Next second:** only 0.046 m of travel but 57° of turn. On a ~0.42 m radius, 57° needs ~0.42 m of travel.
- **Possible causes:** the car bumped a box and was pushed round, the rear wheels slid, someone touched the car, or a gyro glitch.

### 2. Camera identification (`camera_snapshot.py`, new)
- **What it does:** saves one raw picture from each camera ID (0–3) into `camera_snapshots/`, with no calibration files needed. That makes it possible to check which ID is which side before calibrating, because a calibration belongs to whichever camera that ID opens.
- **Result:** cam0 = right, cam1 = rear, cam2 = left, cam3 = front. This matches config.py (`right = "0"`, `left = "2"`) and what the Qcar-master project used.
- **What the right camera's raw picture showed:**
  - **A wide fisheye lens:** straight lines (ceiling lights, window edges) bend toward the corners.
  - **The camera looks straight out sideways,** not down: the horizon is near mid-height. So the **floor only appears in the lower part of the image**, and the bottom of the image is the important region for stall detection.
  - **A pink/purple colour tint,** common for these cameras. Calibration works in black and white, so it doesn't matter there; it may matter later for picking out the tape colour.
  - **A black cable loop was blocking much of the view.** It was rerouted before calibration.

### 3. Lens calibration, right camera (`calibrate_intrinsics.py`)
**Setup:**
- The 9×6-inner-corner checkerboard shown full screen on a 1920×1080 monitor. Each square measured 7/8 inch = **0.0222 m**.
- The car moved by hand so the board appeared in different parts of the camera's view.

| | Attempt 1 | Attempt 2 (`--simple`, 40 frames) |
|---|---|---|
| RMS reprojection error | 0.300 px | 0.430 px |
| Coverage of the image | 56% | 75% (bottom row covered; only the far-left column missed) |
| K: fx, fy, cx, cy | 203.68, 141.79, 210.37, 112.45 | 202.76, 141.02, 210.49, 112.99 |
| D (lens bending terms) | −0.016, **−0.200, +1.042, −1.608** | −0.033, −0.001, 0, 0 |
| Lens model valid out to | sides 71%, top 92%, bottom 96%, **bottom corners 57%** | **all edges and corners** |
| Verdict | Not usable: garbage at the image edges | **Good, in use** |

**What went wrong in attempt 1:** the board only covered the middle of the image, so the fit matched the middle perfectly and made up large, alternating `D` values for the edges. With those values, the lens model "folds back" partway out. Everything beyond that point, including the bottom corners where the floor is, would be distorted wrongly after correction. A low RMS error alone didn't reveal this.

**Why attempt 2 is trustworthy:**
- **Sensible distortion values:** `D` is small.
- **Repeatable:** `K` is almost identical to attempt 1, so the centre and focal length are consistent.
- **Valid everywhere:** the model works to every edge.
- **The missing left column is acceptable:** the fisheye correction depends only on distance from the image centre, and the right side was covered out to the same distances.

**Normal, not a problem:** fx (203) and fy (141) differ. The camera squeezes its sensor's 4:3 picture into the 420×220 frame, so pixels aren't square, and the calibration accounts for that.

### 4. Bird's-eye floor setup: preparation (not yet run)
- **`bev_point_picker.html` (new):** a page that opens in any browser, offline.
  - Load `calib/bev_right_undistorted.png`, then click the centre of each floor marker (shown zoomed 2–6×), and type its `fwd`/`out` position.
  - It downloads `bev_right_points.txt` in the format `make_bev.py solve` expects.
  - Points can be dragged or nudged with the arrow keys. It warns if fewer than 4 points are complete or if `out` isn't positive.
  - **Tested in the browser:** clicks land within a quarter pixel of the target.
- **Grid method for the markers:**
  1. Stick a tape guide line along the outside of the right tyres (parallel to the car).
  2. Mark the floor directly below the lens with a plumb line.
  3. Measure the lens mark's offset from the guide line once (`d0`).
  4. Place 9 crosses at 0.20 / 0.35 / 0.55 m out from the guide line and −0.25 / 0 / +0.25 m along it.

  Each cross's position is then `fwd` = column value, `out` = row distance − `d0`, so no cross needs measuring individually.

---

### Code and tool changes
| File | Change | Why |
|---|---|---|
| `camera_snapshot.py` (new) | Saves one raw picture per camera ID | Confirms which ID is which side before calibrating |
| `parking/intrinsics.py` | Prints a **coverage map** (4×4 grid of where the board was seen) | Shows which parts of the image still need board frames, especially the bottom row where the floor is |
| `parking/intrinsics.py` | New **edge check** (`fisheye_valid_limits`): warns `LENS MODEL FOLDS BACK` with how far toward each edge the model is valid | A low RMS error didn't reveal that attempt 1 was unusable at the edges |
| `parking/intrinsics.py`, `calibrate_intrinsics.py` | New `--simple` option: fixes the two highest lens terms (k3, k4) at 0 | Much harder to break at the edges when the board can't reach every part of the image |
| `parking/intrinsics.py` | Calibration ignores its own `preview_*.png` image | It was being picked up as a calibration frame and skipped with a size warning |
| `bev_point_picker.html` (new) | Click-to-pick page for floor markers | Replaces reading pixel coordinates by hand in Paint |

**On the car or not:** the `preview` skip fix is only on the PC so far. Copy `parking/intrinsics.py` to the car next time files are copied.

---

### Open items / next session
1. **Check the lens result visually:** copy `calib` to the PC and open `calib/intrinsics_frames_right/preview_undistorted.png`. The board's edges in the right half should be straight.
2. **Bird's-eye floor setup, right camera:** car on the floor, lay the marker grid, then:
   - run `make_bev.py snapshot --side right`
   - pick the points with `bev_point_picker.html`
   - run `make_bev.py solve --side right`

   **Expect:** marker errors of a few mm, and crosses inside the yellow circles in `calib/bev_right_preview.png`.
3. **Measure `CAMERA_MOUNT["right"]`:** the lens mark measured forward from the rear axle and out from the car's centre line. It's currently a guess: (0.13, 0.095).
4. **Tape a stall and enter its size:** set `STALL_WIDTH` (line spacing, centre to centre) and `STALL_DEPTH` in config.py. Then tune detection with `calibrate_camera.py --side right`.
5. **Before the first camera-parking run,** measure the odometer scale and the real turn radius (~0.42 m estimated vs 0.47 m in config.py). Without neighbouring obstacles to centre on, the final position depends on both.
6. **Vehicles mode:**
   - Repeat 5 runs with the row at 0.7–0.9 m and a 0.45–0.55 m gap, recording heading error, clearances and success rate.
   - Check whether the mid-turn odometer/heading mismatch happens again.
7. **Still open from 2026-11-02:** LEDs, heading-hold direction, Ctrl+C while moving, `--entry forward`, `--side left`.
