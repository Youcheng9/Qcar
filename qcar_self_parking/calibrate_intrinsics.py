"""Calibrate one side camera's lens (step 1 of camera setup) and write
calib/csi_<side>_intrinsics.yaml.

You need a printed checkerboard, glued flat to cardboard. --board is the number
of INNER corners (where four squares meet), e.g. OpenCV's standard 10x7-square
pattern has 9x6. --square is one square's edge length in meters - measure it.

Usage (on the car):
    python3 calibrate_intrinsics.py --side right --board 9x6 --square 0.025
        captures --frames images while you move the board around, then calibrates
    python3 calibrate_intrinsics.py --side right --board 9x6 --square 0.025 --no-capture
        re-calibrates from the images already saved (also works on a PC without PAL)
    add --simple if the result warns "LENS MODEL FOLDS BACK"

Check the printed coverage map and edge check: the floor is in the BOTTOM of the
image, so the bottom row and bottom corners must be covered.

Next step: make_bev.py
"""

import argparse
import sys

from parking.intrinsics import capture, solve


def board_size(text):
    try:
        cols, rows = (int(v) for v in text.lower().split("x"))
        return cols, rows
    except ValueError:
        raise argparse.ArgumentTypeError("use COLSxROWS of inner corners, e.g. 9x6")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QCar 1 side-camera lens calibration")
    ap.add_argument("--side", choices=["right", "left"], default="right")
    ap.add_argument("--board", type=board_size, default=(9, 6),
                    help="inner corners COLSxROWS (default 9x6)")
    ap.add_argument("--square", type=float, required=True,
                    help="checkerboard square size in meters, e.g. 0.025")
    ap.add_argument("--model", choices=["fisheye", "pinhole"], default="fisheye",
                    help="lens model: fisheye (default, for the QCar's wide-angle CSI cameras) "
                         "or pinhole (narrow lenses)")
    ap.add_argument("--frames", type=int, default=20, help="images to capture (default 20)")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between saved images, time to move the board (default 2)")
    ap.add_argument("--no-capture", action="store_true",
                    help="skip capturing; calibrate from images already saved")
    ap.add_argument("--images-dir", default=None,
                    help="folder of images to calibrate from (default calib/intrinsics_frames_<side>)")
    ap.add_argument("--simple", action="store_true",
                    help="fisheye: fix the two highest lens terms (k3, k4) at 0. Use when the "
                         "result warns that the lens model folds back at the edges")
    args = ap.parse_args()

    if not args.no_capture:
        capture(args.side, args.board, args.frames, args.interval)
    try:
        solve(args.side, args.board, args.square, args.model, args.images_dir, args.simple)
    except RuntimeError as e:
        print("[ERROR] {}".format(e))
        sys.exit(1)
