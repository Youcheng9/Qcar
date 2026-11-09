"""Save bird's-eye frames with detected stall lines from one side camera (car does not move),
for tuning TAPE_HSV_LOW/HIGH and the BEV homography in parking/config.py.

Usage:
    python calibrate_camera.py --side right
"""

import argparse

from parking.calibration import calibrate_camera

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QCar 1 side-camera calibration helper")
    ap.add_argument("--side", choices=["right", "left"], default="right",
                    help="which side camera to capture from")
    calibrate_camera(ap.parse_args().side)
