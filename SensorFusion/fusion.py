"""Shared sensor-fusion sample format: front camera image + 2D LiDAR scan + labels.

No pal/quanser imports, so this runs on the QCar (py3.6) and on a PC.

A session is one folder (one scenario tag per session):
    labels.csv          one row per sample, columns = FIELDS
    images/000123.jpg   front camera frame, BGR as captured
    lidar/000123.npz    raw scan: angles (rad, as the RPLidar reports them), distances (m)

Raw polar scans are stored and turned into points at load time, so cleaning
and orientation (FRONT) can be fixed later without re-collecting.

Car frame for points: x forward, y left, meters.
"""
import csv
import os

import numpy as np

FIELDS = ['index', 't', 'image_t', 'lidar_t', 'image', 'lidar',
          'steering', 'throttle', 'stop', 'tag']

MAX_POINTS = 384     # Lidar(numMeasurements=384) on the QCar
MIN_RANGE = 0.15     # m; RPLidar returns 0 for "no return", and the car body sits inside this
MAX_RANGE = 5.0      # m; indoor track, farther returns are walls. Also the normalization scale.
FRONT = np.pi / 2    # raw LiDAR angle that points straight ahead; raw angles grow clockwise.
                     # Matches TestFusion.py (front = 70..110 deg, left = 60..90 deg).
                     # Verify with lidar_check.py before collecting.


def to_points(angles, distances, max_range=MAX_RANGE):
    """Raw polar scan -> (N, 2) car-frame points (x forward, y left). Drops invalid returns."""
    a = np.asarray(angles, dtype=np.float64).ravel()
    d = np.asarray(distances, dtype=np.float64).ravel()
    n = min(a.size, d.size)
    a, d = a[:n], d[:n]
    ok = np.isfinite(a) & np.isfinite(d) & (d > MIN_RANGE) & (d < max_range)
    theta = FRONT - a[ok]
    return np.stack([d[ok] * np.cos(theta), d[ok] * np.sin(theta)], axis=1)


def pad(points, n=MAX_POINTS, max_range=MAX_RANGE):
    """Normalize points to [-1, 1] and zero-pad (or keep the nearest n) to a fixed size.

    Returns (n, 2) float32 points sorted by bearing, and an (n,) bool mask of real points.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(p) > n:
        p = p[np.argsort(np.hypot(p[:, 0], p[:, 1]))[:n]]
    p = p[np.argsort(np.arctan2(p[:, 1], p[:, 0]))]
    out = np.zeros((n, 2), dtype=np.float32)
    out[:len(p)] = p / max_range
    mask = np.zeros(n, dtype=bool)
    mask[:len(p)] = True
    return out, mask


class SessionWriter:
    """Append samples to a session folder. Flushes every row so a crash keeps the data."""

    def __init__(self, folder, tag):
        self.folder, self.tag, self.index = folder, tag, 0
        os.makedirs(os.path.join(folder, 'images'))
        os.makedirs(os.path.join(folder, 'lidar'))
        self._file = open(os.path.join(folder, 'labels.csv'), 'w', newline='')
        self._csv = csv.DictWriter(self._file, FIELDS)
        self._csv.writeheader()

    def add(self, t, image, image_t, angles, distances, lidar_t, steering, throttle, stop):
        """image / angles may be None (sensor failure); the row is still written."""
        name = '%06d' % self.index
        img = lid = ''
        if image is not None:
            img = 'images/%s.jpg' % name
            cv2_imwrite(os.path.join(self.folder, img), image)
        if angles is not None:
            lid = 'lidar/%s.npz' % name
            np.savez(os.path.join(self.folder, lid),
                     angles=np.asarray(angles, dtype=np.float32),
                     distances=np.asarray(distances, dtype=np.float32))
        self._csv.writerow({
            'index': self.index, 't': '%.4f' % t,
            'image_t': '%.4f' % image_t if image is not None else '',
            'lidar_t': '%.4f' % lidar_t if angles is not None else '',
            'image': img, 'lidar': lid,
            'steering': '%.4f' % steering, 'throttle': '%.4f' % throttle,
            'stop': int(bool(stop)), 'tag': self.tag})
        self._file.flush()
        self.index += 1

    def close(self):
        self._file.close()


def cv2_imwrite(path, image):
    import cv2  # lazy: keeps to_points/pad usable without OpenCV
    if not cv2.imwrite(path, image):
        raise IOError('could not write ' + path)


def read_session(folder):
    """labels.csv -> list of dicts with numeric fields parsed ('' times -> None)."""
    with open(os.path.join(folder, 'labels.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['index'], r['stop'] = int(r['index']), int(r['stop'])
        for k in ('t', 'steering', 'throttle'):
            r[k] = float(r[k])
        for k in ('image_t', 'lidar_t'):
            r[k] = float(r[k]) if r[k] else None
    return rows


def load_scan(folder, row):
    """Raw (angles, distances) for a row, or None if the scan is missing/corrupt."""
    if not row['lidar']:
        return None
    try:
        z = np.load(os.path.join(folder, row['lidar']))
        return z['angles'], z['distances']
    except Exception:
        return None


def load_image(folder, row, size=None):
    """BGR uint8 image for a row, or None if missing/corrupt. size=(w, h) resizes."""
    import cv2
    img = cv2.imread(os.path.join(folder, row['image'])) if row['image'] else None
    if img is not None and size:
        img = cv2.resize(img, size)
    return img


def load(folders, size=(160, 120), n=MAX_POINTS):
    """Load sessions into numpy arrays for training.

    Missing/corrupt sensors stay in as zeros with image_ok / lidar_ok = False,
    so sensor-failure examples are kept and can be filtered or used for dropout.
    """
    if isinstance(folders, str):
        folders = [folders]
    w, h = size
    cols = {k: [] for k in ('image', 'points', 'mask', 'image_ok', 'lidar_ok',
                            'steering', 'throttle', 'stop', 'skew', 'tag')}
    for folder in folders:
        for r in read_session(folder):
            img = load_image(folder, r, size)
            scan = load_scan(folder, r)
            pts, mask = pad(to_points(*scan) if scan else np.zeros((0, 2)), n)
            cols['image'].append(img if img is not None else np.zeros((h, w, 3), np.uint8))
            cols['image_ok'].append(img is not None)
            cols['points'].append(pts)
            cols['mask'].append(mask)
            cols['lidar_ok'].append(bool(mask.any()))
            cols['skew'].append(r['image_t'] - r['lidar_t']
                                if r['image_t'] is not None and r['lidar_t'] is not None else np.nan)
            for k in ('steering', 'throttle', 'stop', 'tag'):
                cols[k].append(r[k])
    out = {k: np.array(v) for k, v in cols.items()}
    out['image'] = out['image'].reshape(-1, h, w, 3)
    out['points'] = out['points'].reshape(-1, n, 2).astype(np.float32)
    out['mask'] = out['mask'].reshape(-1, n)
    return out
