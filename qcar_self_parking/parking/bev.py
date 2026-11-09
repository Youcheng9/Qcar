"""Bird's-eye-view homography setup for one side camera.

snapshot() saves one undistorted PROC_SIZE frame of floor markers you placed,
plus a template points file. solve() reads the points you filled in and
writes calib/csi_<side>_bev.yaml in the format StallLineDetector reads.

Floor coordinates are in meters relative to the camera's GROUND POINT (the spot
on the floor directly below the lens):
    fwd = toward the front of the car (negative = toward the rear)
    out = away from the car, straight out to the side (must be > 0)
"""

import os

import cv2
import numpy as np

from .config import CAMERA_BEV_FILE, PROC_SIZE, SCRIPT_DIR

GRID_STEP_PX = 20               # grid spacing on the *_grid.png helper image
PREVIEW_GRID_M = 0.10           # grid spacing on the bird's-eye preview
MAX_POINT_ERROR_M = 0.02        # warn if a marker reprojects worse than this

POINTS_TEMPLATE = """\
# Bird's-eye calibration points for the {side} camera, one marker per line:
#   u_px  v_px  fwd_m  out_m
# u_px, v_px : pixel column and row of the marker in calib/bev_{side}_undistorted.png
#              ({w}x{h}; open it in Paint/GIMP and read the cursor position, or use
#              the labeled grid in calib/bev_{side}_undistorted_grid.png)
# fwd_m      : meters toward the car's front from the camera's ground point (rear = negative)
# out_m      : meters straight out from the car, from the camera's ground point (> 0)
# At least 4 markers, spread over the area where stall lines will be; not all on one line.
# Example (delete these and use your own measurements):
# 112  150  -0.20  0.30
# 205  148   0.20  0.30
"""


def _paths(side):
    base = os.path.join(SCRIPT_DIR, "calib", "bev_" + side)
    return {"image": base + "_undistorted.png", "grid": base + "_undistorted_grid.png",
            "points": base + "_points.txt", "preview": base + "_preview.png"}


def _grid_image(img):
    vis = img.copy()
    h, w = vis.shape[:2]
    for x in range(0, w, GRID_STEP_PX):
        cv2.line(vis, (x, 0), (x, h - 1), (0, 255, 255), 1)
        if x % (2 * GRID_STEP_PX) == 0:
            cv2.putText(vis, str(x), (x + 1, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)
    for y in range(0, h, GRID_STEP_PX):
        cv2.line(vis, (0, y), (w - 1, y), (0, 255, 255), 1)
        if y % (2 * GRID_STEP_PX) == 0:
            cv2.putText(vis, str(y), (1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 255), 1)
    return cv2.resize(vis, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)   # easier to read


def snapshot(side, warmup_frames=10):
    """Capture one frame, undistort it exactly like the detector does, and save it."""
    import time
    from .camera import SideCamera      # imported here so solve() works without PAL
    from .stall_detector import Undistorter

    undistorter = Undistorter(side)      # fails early with a clear message if intrinsics are missing
    p = _paths(side)
    os.makedirs(os.path.dirname(p["image"]), exist_ok=True)
    camera = SideCamera(side, dedupe=False)
    frame, got = None, 0
    t_end = time.monotonic() + 10.0
    try:
        while got < warmup_frames and time.monotonic() < t_end:   # let exposure settle
            f = camera.read()
            if f is not None:
                frame, got = f.copy(), got + 1
            time.sleep(0.03)
    finally:
        camera.terminate()
    if frame is None:
        raise RuntimeError("no frames received from the {} camera".format(side))

    small = cv2.resize(frame, tuple(PROC_SIZE), interpolation=cv2.INTER_AREA)
    und = undistorter.undistort(small)
    cv2.imwrite(p["image"], und)
    cv2.imwrite(p["grid"], _grid_image(und))
    print("Saved {} ({}x{})".format(p["image"], PROC_SIZE[0], PROC_SIZE[1]))
    print("Saved {} (3x enlarged, grid every {} px, labels are pixel coordinates)".format(
        p["grid"], GRID_STEP_PX))
    if not os.path.isfile(p["points"]):
        with open(p["points"], "w") as fh:
            fh.write(POINTS_TEMPLATE.format(side=side, w=PROC_SIZE[0], h=PROC_SIZE[1]))
        print("Wrote template {} - fill in your markers, then run: make_bev.py solve --side {}"
              .format(p["points"], side))
    else:
        print("Points file {} already exists - check it still matches this snapshot".format(p["points"]))


def solve(side, fwd_range, out_range, mpp):
    """Fit the homography from the points file and write the BEV calibration."""
    p = _paths(side)
    if not os.path.isfile(p["points"]):
        raise RuntimeError("no points file {} - run make_bev.py snapshot first".format(p["points"]))
    try:
        data = np.loadtxt(p["points"], comments="#", ndmin=2)
    except ValueError as e:
        raise RuntimeError("could not read {}: {}".format(p["points"], e))
    if data.size == 0 or data.shape[0] < 4 or data.shape[1] != 4:
        raise RuntimeError("{} needs at least 4 lines of 'u_px v_px fwd_m out_m' (found {})"
                           .format(p["points"], 0 if data.size == 0 else data.shape[0]))
    if np.any(data[:, 3] <= 0):
        raise RuntimeError("every out_m must be > 0 (points under or behind the camera can't be seen)")
    fwd_min, fwd_max = fwd_range
    out_min, out_max = out_range
    if fwd_max <= fwd_min or out_max <= out_min or out_min <= 0 or mpp <= 0:
        raise RuntimeError("bad BEV area: need fwd_min < fwd_max, 0 < out_min < out_max, mpp > 0")

    src = data[:, :2].astype(np.float64)
    floor = data[:, 2:4].astype(np.float64)
    dst = np.column_stack([(floor[:, 0] - fwd_min) / mpp, (floor[:, 1] - out_min) / mpp])
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise RuntimeError("homography fit failed - are 3 or more markers on one straight line?")

    # how well does each marker land where it was measured?
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.hypot(*(proj - dst).T) * mpp
    print("Marker errors after the fit:")
    for (u, v, f, o), e in zip(data, err):
        flag = "  <-- check this one" if e > MAX_POINT_ERROR_M else ""
        print("  px ({:5.1f}, {:5.1f})  floor (fwd {:+.3f}, out {:.3f}) m : {:5.1f} mm{}"
              .format(u, v, f, o, 1000 * e, flag))
    if data.shape[0] == 4:
        print("  (with exactly 4 markers the fit is always exact, so errors can't reveal a "
              "mismeasured marker - 6+ markers recommended)")

    bev_w = int(round((fwd_max - fwd_min) / mpp))
    bev_h = int(round((out_max - out_min) / mpp))
    out_path = CAMERA_BEV_FILE[side]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
    fs.write("H", H)
    fs.write("proc_width", int(PROC_SIZE[0]))
    fs.write("proc_height", int(PROC_SIZE[1]))
    fs.write("bev_width", bev_w)
    fs.write("bev_height", bev_h)
    fs.write("meters_per_px", float(mpp))
    fs.write("origin_fwd_m", float(fwd_min))
    fs.write("origin_out_m", float(out_min))
    fs.release()
    print("Wrote {} (BEV {}x{} px, {:.1f} mm/px, fwd {:+.2f}..{:+.2f} m, out {:.2f}..{:.2f} m)"
          .format(out_path, bev_w, bev_h, 1000 * mpp, fwd_min, fwd_max, out_min, out_max))

    if os.path.isfile(p["image"]):
        _save_preview(cv2.imread(p["image"]), H, bev_w, bev_h, dst, mpp, p["preview"])


def _save_preview(und, H, bev_w, bev_h, dst, mpp, path):
    bev = cv2.warpPerspective(und, H, (bev_w, bev_h))
    step = max(1, int(round(PREVIEW_GRID_M / mpp)))
    for x in range(0, bev_w, step):
        cv2.line(bev, (x, 0), (x, bev_h - 1), (80, 80, 80), 1)
    for y in range(0, bev_h, step):
        cv2.line(bev, (0, y), (bev_w - 1, y), (80, 80, 80), 1)
    for c, r in dst:
        cv2.circle(bev, (int(round(c)), int(round(r))), 4, (0, 255, 255), 1)
    cv2.imwrite(path, bev)
    print("Saved {}: top-down view; the markers should sit inside the yellow circles, "
          "tape lines should look straight and evenly wide, grid = {:.0f} cm. "
          "Left->right = rear->front of the car, top = nearest the car.".format(path, 100 * PREVIEW_GRID_M))
