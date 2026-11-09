"""QCar 1 autonomous perpendicular parking package.

Modules:
    config          all tunable constants (geometry, speeds, gains, thresholds, file paths)
    utils           wrap_deg, LED patterns
    control         SpeedController (PI)
    lidar           LidarView (RPLidar -> car-frame sectors)
    camera          SideCamera (side CSI camera)
    stall_detector  StallLineDetector (taped stall lines from the camera)
    parker          PerpendicularParker (state machine)
    calibration     calibrate (LiDAR), calibrate_camera
    intrinsics      lens calibration from checkerboard images (calibrate_intrinsics.py)
    bev             bird's-eye homography from floor markers (make_bev.py)
"""
