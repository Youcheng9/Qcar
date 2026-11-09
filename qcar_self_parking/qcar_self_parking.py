"""
QCar 1 - Autonomous Perpendicular Parking
=========================================

Two modes:

  --mode vehicles   Parking next to parked vehicles.
                    Drives forward along a row of parked cars (noses facing the
                    aisle), uses the side LiDAR beam to find a gap wide enough
                    for the QCar, computes the spot center, and parks in it.
                    While entering, it uses the neighbors on both sides to stay
                    centered.

  --mode empty      Parking in an empty parking lot (no reference vehicles).
                    Drives forward looking at the floor with the side CSI camera
                    on --side, detects taped stall lines (undistort -> bird's-eye
                    view -> color threshold -> HoughLinesP), finds an adjacent
                    pair of lines spaced STALL_WIDTH apart, confirms it over
                    several frames, then parks in that stall. LiDAR is still used
                    for safety and to stop at a curb/wall at the back of the spot.

Entry direction (both modes):  --entry reverse (default) | forward
Spot side (both modes):        --side right (default) | left

State machine:
    GYRO_CAL  (both)     hold still ~2 s, estimate gyro z bias
    SEARCH    (vehicles) creep forward, find a LiDAR gap of sufficient width
              (empty)    creep forward, find a camera-detected stall
    POSITION  drive straight forward/back to the turn start point
    TURN      90 deg arc at full steering lock toward the spot
    ENTER     straight into the spot with heading hold (+ centering in vehicles mode)
    DONE      stop, hazards on
    ABORT     refused / gave up (reason printed), stop, hazards on

Geometry (rear-axle point follows a circle of radius R = WHEELBASE / tan(MAX_STEER)):
    reverse-in: start the turn with the rear axle R *past* the spot center
    forward-in: start the turn with the rear axle R *before* the spot center
    after the 90 deg arc the rear axle sits R deep (laterally) from the start line.

Hardware / software:
    Quanser QCar 1 (or virtual QCar in QLabs), Quanser PAL python library, OpenCV.
    Sensors: RPLidar (QCarLidar), motor tachometer, IMU gyroscope, side CSI camera.

PAL API assumptions (NOT verified against your PAL version - check them):
    - QCar.motorTach is the car's linear speed in m/s, signed (+ forward).
      Some PAL versions/QCar models expose motor-shaft speed instead; if so the
      odometry will be scaled wrong. See "encoder scale" in the measurement list.
    - QCar.gyroscope is [x, y, z] in rad/s.
    - QCarLidar.read() may block until a new scan arrives (~10 Hz on the RPLidar)
      or may return immediately with the last scan, depending on PAL version.
      The loop integrates with measured dt, so either behavior is handled; the
      once-per-second timing log shows which one you have.
    - QCar(readMode=0) does immediate (non-task-based) I/O on each read_write_std.
    - PAL's QCar class works on a QCar 1: the Qcar-master programs drive one with
      QCar(readMode=0) and read_write_std(throttle, steering, LEDs) (+ steering =
      left; LEDs 0,2 = left signal, 1,3 = right, 4 = brake, 5 = reverse, 6,7 = head).
      They never read motorTach or the gyroscope, so those stay unverified.
    - Camera2D(cameraId, frameWidth, frameHeight, frameRate) from
      pal.utilities.vision, as in Qcar-master (IDs 0 right, 1 back, 2 left,
      3 front; 420x220 @ 30 fps). .read() refreshes .imageData (HxWx3 uint8, BGR).
      The return value of read() and whether it blocks are not verified;
      duplicate frames are filtered out. Confirm with --calibrate-camera that
      "right" really shows the right side (CAMERA_ID in parking/config.py).

Camera calibration files (per side, OpenCV FileStorage YAML, paths in config).
Empty mode needs both. Create them in this order:
    1. calibrate_intrinsics.py  -> calib/csi_<side>_intrinsics.yaml (checkerboard)
    2. make_bev.py snapshot / solve -> calib/csi_<side>_bev.yaml (floor markers)
    3. measure CAMERA_MOUNT in parking/config.py
    4. calibrate_camera.py to check detection and tune the tape thresholds
    intrinsics:  model ("pinhole" | "fisheye"), image_width, image_height
                 (= the capture size the calibration was done at), K (3x3), D.
    bird's-eye:  H (3x3) mapping UNDISTORTED pixels at PROC_SIZE resolution
                 (exactly what make_bev.py snapshot saves) to BEV pixels;
                 proc_width, proc_height; bev_width, bev_height;
                 meters_per_px; origin_fwd_m, origin_out_m = floor position of
                 BEV pixel (0,0) relative to the camera's ground point.
                 BEV convention: column -> +forward, row -> +outward (away from
                 the car), so row 0 is the edge nearest the car.

Project layout:
    qcar_self_parking.py   this script: main entry point (all modes / flags)
    calibrate_lidar.py     shortcut for --calibrate
    calibrate_camera.py    shortcut for --calibrate-camera
    calibrate_intrinsics.py  camera lens calibration (camera setup step 1)
    make_bev.py            camera bird's-eye view setup (camera setup step 2)
    parking/              the package (see parking/__init__.py); tune values in parking/config.py

Usage:
    python qcar_self_parking.py --calibrate          (or: python calibrate_lidar.py)
    python qcar_self_parking.py --calibrate-camera --side right
                                                     (or: python calibrate_camera.py --side right)
    python qcar_self_parking.py --mode vehicles
    python qcar_self_parking.py --mode vehicles --entry forward --side left
    python qcar_self_parking.py --mode empty --entry reverse --side right

ALWAYS test with the car on a stand first, then at low speed with a hand on
the e-stop. Ctrl+C stops the car immediately.
"""

import argparse
import sys

from parking.calibration import calibrate, calibrate_camera
from parking.parker import PerpendicularParker


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QCar 1 autonomous perpendicular parking")
    ap.add_argument("--mode", choices=["vehicles", "empty"], default="vehicles",
                    help="vehicles: find a gap between parked cars (LiDAR); "
                         "empty: find a taped stall in an empty lot (side camera)")
    ap.add_argument("--entry", choices=["reverse", "forward"], default="reverse",
                    help="back into the spot (default) or drive in nose first")
    ap.add_argument("--side", choices=["right", "left"], default="right",
                    help="which side of the aisle the spot is on (also selects the camera)")
    ap.add_argument("--calibrate", action="store_true",
                    help="print LiDAR sector distances without moving the car")
    ap.add_argument("--calibrate-camera", action="store_true",
                    help="save bird's-eye frames with detected lines for the --side camera "
                         "without moving the car")
    args = ap.parse_args()

    if args.calibrate:
        calibrate()
    elif args.calibrate_camera:
        calibrate_camera(args.side)
    else:
        PerpendicularParker(args.mode, args.entry, args.side).run()
    sys.exit(0)
