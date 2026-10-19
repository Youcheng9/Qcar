"""Check and step through recorded fusion sessions (run on the PC or the car).

    python3 view_session.py data/20261020_101500_normal            # summary + viewer
    python3 view_session.py data/*  --summary                       # summary only (no window)

Viewer keys: space play/pause, d / a next / previous, q or ESC quit.
"""
import argparse
import os

import cv2
import numpy as np

import fusion

BEV = 300                       # bird's-eye view size in px; car at bottom-center, forward = up
SCALE = BEV / fusion.MAX_RANGE  # px per meter


def summary(folder):
    rows = fusion.read_session(folder)
    n = len(rows)
    print('\n%s: %d samples' % (folder, n))
    if not n:
        return rows
    no_img = sum(1 for r in rows if fusion.load_image(folder, r) is None)
    scans = [fusion.load_scan(folder, r) for r in rows]
    valid = [len(fusion.to_points(*s)) if s else 0 for s in scans]
    skew = np.array([r['image_t'] - r['lidar_t'] for r in rows
                     if r['image_t'] is not None and r['lidar_t'] is not None])
    steer = np.array([r['steering'] for r in rows])
    dur = rows[-1]['t'] - rows[0]['t']
    print('  duration %.0fs (%.1f Hz)   tag(s) %s' % (dur, (n - 1) / dur if dur else 0,
                                                     sorted({r['tag'] for r in rows})))
    print('  missing/corrupt images %d   missing scans %d   scans with <50 valid pts %d' % (
        no_img, sum(s is None for s in scans), sum(v < 50 for v in valid)))
    print('  valid LiDAR points/scan  mean %.0f  min %d' % (np.mean(valid), min(valid)))
    if skew.size:
        print('  camera-LiDAR skew  median %.3fs  max %.3fs  >0.25s: %d' % (
            np.median(skew), skew.max(), int((skew > 0.25).sum())))
    print('  stop samples %d   steering  left %d / straight %d / right %d' % (
        sum(r['stop'] for r in rows), (steer > 0.05).sum(), (abs(steer) <= 0.05).sum(),
        (steer < -0.05).sum()))
    return rows


def bev(points):
    img = np.full((BEV, BEV, 3), 30, np.uint8)
    for r in range(1, int(fusion.MAX_RANGE) + 1):
        cv2.circle(img, (BEV // 2, BEV), int(r * SCALE), (70, 70, 70), 1)
    for x, y in points:
        cv2.circle(img, (int(BEV / 2 - y * SCALE), int(BEV - x * SCALE)), 2, (0, 255, 0), -1)
    cv2.rectangle(img, (BEV // 2 - 5, BEV - 12), (BEV // 2 + 5, BEV), (0, 128, 255), -1)
    return img


def frame(folder, r):
    img = fusion.load_image(folder, r)
    img = cv2.resize(img, (420, 220)) if img is not None else np.zeros((220, 420, 3), np.uint8)
    img = cv2.copyMakeBorder(img, 0, BEV - 220, 0, 0, cv2.BORDER_CONSTANT)
    scan = fusion.load_scan(folder, r)
    pts = fusion.to_points(*scan) if scan else np.zeros((0, 2))
    skew = r['image_t'] - r['lidar_t'] if r['image_t'] and r['lidar_t'] else float('nan')
    lines = ['#%d  %s' % (r['index'], r['tag']),
             'steer %+.3f  throttle %+.3f' % (r['steering'], r['throttle']),
             'skew %.3fs  lidar pts %d' % (skew, len(pts)),
             'STOP' if r['stop'] else '']
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 240 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if s == 'STOP' else (255, 255, 255), 1)
    # steering arrow: positive steering = left on the QCar
    c = (210, 215)
    cv2.arrowedLine(img, c, (int(c[0] - 60 * np.sin(r['steering'] * 2)),
                             int(c[1] - 60 * np.cos(r['steering'] * 2))), (0, 255, 255), 2)
    return np.hstack([img, bev(pts)])


def view(folder, rows):
    i, play = 0, False
    while rows:
        cv2.imshow('fusion', frame(folder, rows[i]))
        k = cv2.waitKey(100 if play else 0) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord(' '):
            play = not play
        elif k == ord('a'):
            i = max(i - 1, 0)
        elif k == ord('d') or play:
            i = min(i + 1, len(rows) - 1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folders', nargs='+')
    p.add_argument('--summary', action='store_true', help='print stats only')
    a = p.parse_args()
    for f in a.folders:
        if os.path.isfile(os.path.join(f, 'labels.csv')):
            rows = summary(f)
            if not a.summary:
                view(f, rows)
