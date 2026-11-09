"""Print LiDAR sector distances (car does not move) to confirm FRONT_ANGLE_DEG / ANGLE_DIR
in parking/config.py.

Usage:
    python calibrate_lidar.py
"""

from parking.calibration import calibrate

if __name__ == "__main__":
    calibrate()
