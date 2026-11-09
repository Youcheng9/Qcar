"""Set up one side camera's bird's-eye (top-down) view (step 2 of camera setup) and write
calib/csi_<side>_bev.yaml. Needs calib/csi_<side>_intrinsics.yaml from calibrate_intrinsics.py.

1. Mark the camera's ground point: the spot on the floor directly below the lens.
2. Put at least 4 (6+ is better) small markers (tape crosses) on the floor in view of
   the camera, spread over the area where stall lines will be. Measure each one from the
   ground point: fwd = toward the car's front (rear = negative), out = straight away from the car.
3. On the car (do not move it or the markers):
       python3 make_bev.py snapshot --side right
   This saves calib/bev_right_undistorted.png (+ a _grid.png helper) and a template
   calib/bev_right_points.txt.
4. Copy the images to your PC if needed, read each marker's pixel position, and fill in
   bev_right_points.txt as "u_px v_px fwd_m out_m" lines.
5. Fit and save the homography (works on the car or on a PC):
       python3 make_bev.py solve --side right
   Then check calib/bev_right_preview.png.

Next: measure CAMERA_MOUNT in parking/config.py, then run calibrate_camera.py.
"""

import argparse
import sys

from parking.bev import snapshot, solve


def float_range(text):
    try:
        lo, hi = (float(v) for v in text.split(","))
        return lo, hi
    except ValueError:
        raise argparse.ArgumentTypeError("use MIN,MAX in meters, e.g. -0.5,0.5")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QCar 1 side-camera bird's-eye view setup")
    ap.add_argument("step", choices=["snapshot", "solve"],
                    help="snapshot: save an undistorted frame of your floor markers; "
                         "solve: fit the homography from the points file")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--fwd", type=float_range, default=(-0.5, 0.5),
                    help="solve: floor area covered along the car, MIN,MAX meters from the "
                         "camera ground point (default -0.5,0.5)")
    ap.add_argument("--out", type=float_range, default=(0.05, 0.85),
                    help="solve: floor area covered away from the car, MIN,MAX meters, MIN > 0 "
                         "(default 0.05,0.85)")
    ap.add_argument("--mpp", type=float, default=0.005,
                    help="solve: meters per bird's-eye pixel (default 0.005 = 5 mm)")
    args = ap.parse_args()

    try:
        if args.step == "snapshot":
            snapshot(args.side)
        else:
            solve(args.side, args.fwd, args.out, args.mpp)
    except (RuntimeError, FileNotFoundError, ValueError, IOError) as e:
        print("[ERROR] {}".format(e))
        sys.exit(1)
