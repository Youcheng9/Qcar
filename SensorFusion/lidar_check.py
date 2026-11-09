"""Run on the QCar before collecting: confirms fusion.FRONT / angle direction.

Put a box ~0.5 m straight in front of the car, then ~0.5 m to its left.
The nearest point should read x > 0, y ~ 0 (front), then y > 0 (left).
If not, fix FRONT (or the sign in fusion.to_points) -- raw scans are stored, so
already-collected sessions are fixed by the same edit.

    sudo python3 lidar_check.py
"""
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'LidarStuff'))
from lidar2 import Lidar
import fusion

lidar = Lidar(type='RPLidar')
try:
    while True:
        lidar.read()
        pts = fusion.to_points(lidar.angles, lidar.distances)
        raw = np.asarray(lidar.distances).ravel()
        if len(pts):
            x, y = pts[np.argmin(np.hypot(pts[:, 0], pts[:, 1]))]
            side = 'FRONT' if x > abs(y) else 'BACK' if -x > abs(y) else 'LEFT' if y > 0 else 'RIGHT'
            print('valid %3d/%3d  nearest %.2f m  x %+.2f y %+.2f  -> %s' % (
                len(pts), raw.size, np.hypot(x, y), x, y, side))
        else:
            print('no valid points (%d raw) -- LiDAR spinning? cable?' % raw.size)
        time.sleep(0.3)
except KeyboardInterrupt:
    pass
finally:
    lidar.terminate()
