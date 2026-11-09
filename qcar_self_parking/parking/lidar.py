"""RPLidar wrapper giving distances in car-frame sectors."""

import numpy as np

from pal.products.qcar import QCarLidar

from .config import (ANGLE_DIR, FRONT_ANGLE_DEG, LIDAR_MAX_VALID, LIDAR_MIN_VALID,
                     SAFETY_SECTOR_HALF_WIDTH, SIDE_SECTOR_HALF_WIDTH)
from .utils import wrap_deg


class LidarView:
    """Wraps QCarLidar and gives distances in car-frame sectors.
    Car-frame angle: 0 = front, +90 = left, -90 = right, 180 = rear."""

    def __init__(self):
        # enableFiltering=False: PAL's filter_rplidar_data() drops invalid (zero)
        # reads with np.delete(angles, distances == 0). On the car's older numpy a
        # boolean index is cast to integers, so it deletes elements 0 and 1 instead,
        # keeps the zeros, and then interpolates between zeros and real ranges -
        # random 0.02-0.5 m readings in every direction. The raw scan has the same
        # angle convention (that filter only resamples), and update() drops the
        # zeros itself.
        self.lidar = QCarLidar(numMeasurements=720,
                               rangingDistanceMode=2,
                               interpolationMode=0,
                               enableFiltering=False)
        self.car_angles = None
        self.dist = None

    def update(self):
        # Returns immediately with the latest scan (loop timing stays ~10 ms).
        self.lidar.read()
        raw_deg = np.degrees(np.asarray(self.lidar.angles, dtype=float).reshape(-1))
        self.car_angles = wrap_deg(ANGLE_DIR * (raw_deg - FRONT_ANGLE_DEG))
        d = np.asarray(self.lidar.distances, dtype=float).reshape(-1).copy()
        d[(d < LIDAR_MIN_VALID) | (d > LIDAR_MAX_VALID)] = np.inf
        self.dist = d

    def sector(self, center_deg, half_width_deg, stat="min"):
        if self.dist is None:
            return np.inf
        diff = np.abs(wrap_deg(self.car_angles - center_deg))
        vals = self.dist[diff <= half_width_deg]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.inf
        return float(np.min(vals) if stat == "min" else np.median(vals))

    def nearest(self, center_deg, half_width_deg):
        """(distance, car-frame angle) of the closest return in the sector, or (inf, nan)."""
        if self.dist is None:
            return np.inf, np.nan
        diff = np.abs(wrap_deg(self.car_angles - center_deg))
        idx = np.where((diff <= half_width_deg) & np.isfinite(self.dist))[0]
        if idx.size == 0:
            return np.inf, np.nan
        i = idx[np.argmin(self.dist[idx])]
        return float(self.dist[i]), float(self.car_angles[i])

    def front(self):
        return self.sector(0.0, SAFETY_SECTOR_HALF_WIDTH)

    def rear(self):
        return self.sector(180.0, SAFETY_SECTOR_HALF_WIDTH)

    def side(self, side_sign):
        """side_sign +1 = right, -1 = left"""
        return self.sector(-90.0 * side_sign, SIDE_SECTOR_HALF_WIDTH, stat="median")

    def terminate(self):
        self.lidar.terminate()
