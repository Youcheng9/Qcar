"""Save one raw frame from each CSI camera (car does not move), to check which
cameraId is which side before calibrating. No calibration files needed.

Usage (on the car):
    sudo python3 camera_snapshot.py            # all four IDs 0-3
    sudo python3 camera_snapshot.py --ids 0 2  # only some

Saves camera_snapshots/cam<ID>.png. Open them and check that the IDs in
CAMERA_ID (parking/config.py) show the side they claim: right = "0", left = "2".
"""

import argparse
import os
import time

import cv2

from pal.utilities.vision import Camera2D

from parking.config import CAMERA_CAPTURE_SIZE, CAMERA_FPS, CAMERA_ID, SCRIPT_DIR

OUT_DIR = os.path.join(SCRIPT_DIR, "camera_snapshots")
WARMUP_S = 1.5                  # let auto-exposure settle before saving


def snapshot(cam_id):
    cam = Camera2D(cameraId=str(cam_id), frameWidth=CAMERA_CAPTURE_SIZE[0],
                   frameHeight=CAMERA_CAPTURE_SIZE[1], frameRate=CAMERA_FPS)
    try:
        t_end = time.monotonic() + WARMUP_S
        while time.monotonic() < t_end:
            cam.read()
            time.sleep(0.03)
        cam.read()
        img = cam.imageData
    finally:
        cam.terminate()
    if img is None or getattr(img, "size", 0) == 0:
        print("  cameraId {}: no image".format(cam_id))
        return
    path = os.path.join(OUT_DIR, "cam{}.png".format(cam_id))
    cv2.imwrite(path, img)
    print("  cameraId {}: saved {} ({}x{})".format(cam_id, path, img.shape[1], img.shape[0]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Save one raw frame per CSI camera")
    ap.add_argument("--ids", nargs="+", default=["0", "1", "2", "3"], help="camera IDs (default 0 1 2 3)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("CAMERA_ID in config.py: right = {right!r}, left = {left!r}".format(**CAMERA_ID))
    for cam_id in args.ids:
        try:
            snapshot(cam_id)
        except Exception as e:      # one bad camera shouldn't stop the others
            print("  cameraId {}: failed ({})".format(cam_id, e))
    print("Open the images: the 'right' ID must show the view out of the car's right side.")
