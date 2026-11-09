"""Calibration helpers that do not move the car: LiDAR angle check and camera frame dump."""

import os
import time

import cv2

from .camera import SideCamera
from .config import (CAMCAL_INTERVAL, CAMCAL_NUM_FRAMES, CAMCAL_OUTPUT_DIR,
                     LIDAR_TO_FRONT_BUMPER, LIDAR_TO_REAR_BUMPER, SAFETY_SECTOR_HALF_WIDTH,
                     SAFETY_STOP_DIST)
from .lidar import LidarView
from .stall_detector import StallLineDetector


def calibrate():
    """Print sector distances so you can confirm FRONT_ANGLE_DEG / ANGLE_DIR, plus the
    closest return in each safety sector (what makes the car hold) and all around."""
    lidar = LidarView()
    front_limit = LIDAR_TO_FRONT_BUMPER + SAFETY_STOP_DIST
    rear_limit = LIDAR_TO_REAR_BUMPER + SAFETY_STOP_DIST
    print("Hold an object at the front / right / rear / left of the car. Ctrl+C to quit.\n"
          "safety = closest return within +/-{:.0f} deg; STOP means the car would hold "
          "(front < {:.2f} m, rear < {:.2f} m)".format(SAFETY_SECTOR_HALF_WIDTH, front_limit, rear_limit))
    try:
        while True:
            lidar.update()
            fd, fa = lidar.nearest(0.0, SAFETY_SECTOR_HALF_WIDTH)
            rd, ra = lidar.nearest(180.0, SAFETY_SECTOR_HALF_WIDTH)
            cd, ca = lidar.nearest(0.0, 180.0)
            print(f"front {lidar.sector(0, 10):5.2f}  "
                  f"left {lidar.sector(90, 10):5.2f}  "
                  f"rear {lidar.sector(180, 10):5.2f}  "
                  f"right {lidar.sector(-90, 10):5.2f} [m] | safety: "
                  f"front {fd:4.2f}@{fa:+4.0f} {'STOP' if fd < front_limit else 'ok  '}  "
                  f"rear {rd:4.2f}@{ra:+4.0f} {'STOP' if rd < rear_limit else 'ok  '} | "
                  f"closest {cd:4.2f}@{ca:+4.0f}   ", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()
    finally:
        lidar.terminate()


def calibrate_camera(side):
    """Save sample frames (car does not move) so tape thresholds / BEV can be tuned:
         NN_undistorted.png  undistorted PROC_SIZE frame (use this to pick homography points)
         NN_bev_overlay.png  bird's-eye view with red = all Hough segments,
                             green = perpendicular segments, blue = merged tape lines
                             (cyan dot = entrance end visible, red dot = cut off),
                             yellow circle = detected stall center
         NN_mask.png         tape threshold mask in the bird's-eye view
    """
    detector = StallLineDetector(side)
    camera = SideCamera(side, dedupe=False)
    os.makedirs(CAMCAL_OUTPUT_DIR, exist_ok=True)
    print(f"Saving {CAMCAL_NUM_FRAMES} frames from the {side} camera to {CAMCAL_OUTPUT_DIR} "
          f"every {CAMCAL_INTERVAL:.1f} s. The car will not move. Ctrl+C to stop.")
    saved = 0
    next_t = time.monotonic()
    try:
        while saved < CAMCAL_NUM_FRAMES:
            frame = camera.read()
            if frame is None or time.monotonic() < next_t:
                time.sleep(0.01)
                continue
            next_t += CAMCAL_INTERVAL
            det = detector.detect(frame)
            base = os.path.join(CAMCAL_OUTPUT_DIR, f"{saved:02d}")
            cv2.imwrite(base + "_undistorted.png", detector.undistort(det["small"]))
            cv2.imwrite(base + "_bev_overlay.png", detector.draw(det))
            cv2.imwrite(base + "_mask.png", det["mask"])
            lines = ", ".join(f"{c['fwd']:+.2f}{'' if c['entrance_visible'] else '(cut)'}"
                              for c in det["clusters"]) or "none"
            stalls = ", ".join(f"(fwd {s['fwd']:+.2f}, lateral {s['lateral']:.2f}, "
                               f"spacing {s['spacing']:.2f})" for s in det["stalls"]) or "none"
            print(f"[{saved:02d}] segments {len(det['segs'])}, perpendicular {len(det['kept'])} | "
                  f"lines at fwd [m]: {lines} | stalls: {stalls}")
            saved += 1
    except KeyboardInterrupt:
        print()
    finally:
        camera.terminate()
