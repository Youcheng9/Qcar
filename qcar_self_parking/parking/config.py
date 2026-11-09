"""Configuration  (tune these for your car / track)."""

import math
import os

# Project root (the folder containing qcar_self_parking.py), so calibration
# files and output folders live in the same place as before the split.
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- QCar 1 geometry [m] ---
CAR_LENGTH = 0.425
CAR_WIDTH = 0.192
WHEELBASE = 0.256
REAR_OVERHANG = 0.085          # rear bumper -> rear axle
LIDAR_FROM_REAR_AXLE = 0.13    # lidar is this far ahead of the rear axle
MAX_STEER = 0.5                # [rad] QCar steering limit (+ = left)

# derived
TURN_RADIUS = WHEELBASE / math.tan(MAX_STEER)                 # ~0.47 m (verify by measurement)
CENTER_FROM_REAR_AXLE = CAR_LENGTH / 2.0 - REAR_OVERHANG      # geometric center offset
LIDAR_TO_FRONT_BUMPER = CAR_LENGTH - REAR_OVERHANG - LIDAR_FROM_REAR_AXLE
LIDAR_TO_REAR_BUMPER = REAR_OVERHANG + LIDAR_FROM_REAR_AXLE

# --- Control loop / timing ---
SAMPLE_RATE = 100              # [Hz] target rate; real rate may be lower if lidar.read() blocks
DT = 1.0 / SAMPLE_RATE         # target period (used only for sleeping, not for integration)
DT_MAX = 0.2                   # [s] clamp on measured dt so a hiccup can't cause a big jump
TIMING_LOG_PERIOD = 1.0        # [s] how often to print average / max loop time
MAX_RUN_TIME = 120.0           # [s] hard timeout

# --- Gyro bias calibration (car stationary, zero throttle) ---
GYRO_CAL_DURATION = 15.0       # [s] averaging window. Stand test: 5 s left ~0.5 deg/s heading
                               # drift, 15 s brought it to ~0.04 deg/s (the gyro is noisy)
GYRO_CAL_MAX_STD = 0.15        # [rad/s] above this the car was probably bumped/moving.
                               # gyro_check.py measured ~0.09 rad/s at rest on this QCar 1
                               # (was 0.01, which this sensor can never pass)
GYRO_CAL_MAX_ATTEMPTS = 3      # give up (abort) after this many noisy attempts

# --- Speeds [m/s] ---
SEARCH_SPEED = 0.20            # vehicles-mode LiDAR search
STALL_SEARCH_SPEED = 0.12      # empty-mode camera search (slower: less blur, more frames per stall)
POSITION_SPEED = 0.15
TURN_SPEED = 0.12
ENTER_SPEED = 0.08

# --- Speed PI controller (throttle command is roughly -1..1) ---
KP_SPEED = 0.15
KI_SPEED = 0.8
MAX_THROTTLE = 0.20            # keep this low while testing

# --- Heading hold / centering while entering the spot ---
K_HEADING = 1.5                # [rad steer / rad heading error]
K_CENTER = 1.0                 # [rad steer / m lateral error] (vehicles mode)
MAX_CORRECTION_STEER = 0.20
TURN_ARC_TOLERANCE = 1.3       # abort arc if odometry exceeds 1.3x expected arc length

# --- LiDAR angle convention ---
# Run calibrate_lidar.py and hold your hand in front of / right of the car to check.
# FRONT_ANGLE_DEG: raw lidar angle that points straight ahead.
# ANGLE_DIR: +1 if raw angles increase counter-clockwise (toward left), -1 if clockwise.
FRONT_ANGLE_DEG = 90.0         # was 0.0: every direction read 90 deg clockwise of the truth
ANGLE_DIR = -1                 # left/right order was correct, so unchanged
LIDAR_MAX_VALID = 5.0          # [m] readings beyond this are treated as "no return"
LIDAR_MIN_VALID = 0.05         # [m] readings below this (incl. the 0 = invalid reads) are dropped;
                               # nothing real can be closer than the LiDAR's own housing

# --- Gap detection (vehicles mode) ---
SIDE_SECTOR_HALF_WIDTH = 3.0   # [deg] around the 90 deg side beam
GAP_DEPTH_JUMP = 0.20          # [m] side distance must grow by this much to count as a gap
SPOT_SIDE_MARGIN = 0.10        # [m] required clearance on each side of the car
MIN_GAP_WIDTH = CAR_WIDTH + 2 * SPOT_SIDE_MARGIN   # ~0.39 m
MAX_GAP_WIDTH = 3.0 * CAR_WIDTH                    # wider than this = open space; use first part
ENTRANCE_MARGIN = 0.03         # [m] how far past the neighbors' bumpers to pull in

# --- Search limits (both modes) ---
MAX_SEARCH_DISTANCE = 3.0      # [m] give up if no usable gap / stall found within this distance
ROW_DETECT_MAX_DIST = 1.5      # [m] (vehicles) side return closer than this counts as "a parked car"

# --- Row clearance guard (vehicles mode) ---
# The single-arc maneuver must finish turning before the leading bumper crosses
# the neighbors' bumper line. Leading-bumper depth at the end of the arc:
#   reverse-in: TURN_RADIUS + REAR_OVERHANG
#   forward-in: TURN_RADIUS + (CAR_LENGTH - REAR_OVERHANG)
# The car refuses to park if the row is closer than that depth + this margin.
ROW_CLEARANCE_MARGIN = 0.05    # [m]

# --- Camera capture / preprocessing (empty mode) ---
# Camera2D cameraId per side. The Qcar-master programs use 0 = right, 1 = back,
# 2 = left, 3 = front. Check with calibrate_camera.py that "right" shows the right side.
CAMERA_ID = {
    "right": "0",
    "left": "2",
}
# (w, h) requested from the CSI camera. 420x220 @ 30 fps is what the Qcar-master
# programs used. Changing it means redoing calibrate_intrinsics.py AND make_bev.py.
CAMERA_CAPTURE_SIZE = (420, 220)
CAMERA_FPS = 30
CAMERA_COLOR_ORDER = "BGR"         # "BGR" or "RGB" - Qcar-master treats Camera2D.imageData as BGR
PROC_SIZE = CAMERA_CAPTURE_SIZE    # (w, h) frames are resized to this before undistort/BEV.
                                   # Set a smaller (w, h) to save CPU at large capture sizes;
                                   # K is scaled per axis. Changing it means redoing make_bev.py.
CAMERA_INTRINSICS_FILE = {
    "right": os.path.join(SCRIPT_DIR, "calib", "csi_right_intrinsics.yaml"),
    "left": os.path.join(SCRIPT_DIR, "calib", "csi_left_intrinsics.yaml"),
}
CAMERA_BEV_FILE = {
    "right": os.path.join(SCRIPT_DIR, "calib", "csi_right_bev.yaml"),
    "left": os.path.join(SCRIPT_DIR, "calib", "csi_left_bev.yaml"),
}
FISHEYE_BALANCE = 0.0          # fisheye model: 0 = crop to valid pixels, 1 = keep full FOV
FISHEYE_FALLBACK_FOCAL_SCALE = 0.6  # used instead when OpenCV's estimate fails (very wide lens):
                                    # < 1 shows more of the floor at lower resolution.
                                    # Changing either fisheye value means redoing make_bev.py.
PINHOLE_ALPHA = 0.0            # pinhole model: same idea, for getOptimalNewCameraMatrix
# Camera ground point (point on the floor directly below the lens) in the car frame:
#   (forward of rear axle [m], outward from car centerline toward that side [m]).  MEASURE.
CAMERA_MOUNT = {
    "right": (0.13, 0.095),
    "left": (0.13, 0.095),
}
CAMERA_LATENCY_S = 0.0         # [s] capture -> processing delay; odom is back-dated by speed*latency

# --- Stall-line detection (empty mode) ---
TAPE_HSV_LOW = (0, 0, 180)     # default: white/bright tape. Tune with calibrate_camera.py.
TAPE_HSV_HIGH = (180, 60, 255)
MORPH_KERNEL_PX = 3            # opening kernel to remove speckle
HOUGH_THRESHOLD = 20           # accumulator votes
HOUGH_MIN_LINE_LENGTH = 0.06   # [m] shortest segment HoughLinesP returns
HOUGH_MAX_LINE_GAP = 0.03      # [m] gap HoughLinesP may bridge
LINE_ANGLE_TOL_DEG = 15.0      # keep segments within this of perpendicular to travel
LINE_MERGE_DIST = 0.04         # [m] segments closer than this along the aisle = same tape line
MIN_LINE_EXTENT = 0.10         # [m] a tape line must be visible over at least this lateral length
ENTRANCE_EDGE_MARGIN_PX = 3    # line's near end must be this far from the BEV's near valid edge
                               # (otherwise the stall entrance is cut off and can't be located)
ENTRANCE_ALIGN_TOL = 0.05      # [m] the two lines' near ends must agree this well
STALL_WIDTH = 0.40             # [m] center-to-center spacing of the two stall lines.  MEASURE.
STALL_WIDTH_TOL = 0.05         # [m]
STALL_DEPTH = 0.60             # [m] stall length from entrance to back.  MEASURE.
STALL_CONFIRM_FRAMES = 5       # consecutive frames the same stall must be seen before committing
STALL_MAX_MISSES = 0           # frames without a match tolerated inside a run (0 = strictly consecutive)
STALL_MATCH_TOL = 0.05         # [m] along-aisle agreement for "same stall" between frames
STALL_MATCH_TOL_LATERAL = 0.05 # [m] lateral agreement for "same stall" between frames

# --- calibrate_camera.py helper ---
CAMCAL_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "camera_calib_frames")
CAMCAL_NUM_FRAMES = 10
CAMCAL_INTERVAL = 1.0          # [s] between saved frames

# --- Safety ---
SAFETY_STOP_DIST = 0.08        # [m] bumper clearance that triggers a stop
SAFETY_SECTOR_HALF_WIDTH = 15.0
SAFETY_LOG_PERIOD = 0.5        # [s] at most one [SAFETY] message per this period

# --- Heading sign ---
# +1 if gyroscope[2] is positive when the car turns LEFT (CCW). Verify on a stand.
GYRO_Z_SIGN = 1

# --- LEDs (QCar has 8). Verify indices on your car. ---
LED_LEFT_IND = (0, 2)
LED_RIGHT_IND = (1, 3)
LED_BRAKE = (4, 5)
LED_HEAD = (6, 7)
