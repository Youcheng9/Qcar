"""Lens (intrinsics) calibration from checkerboard images.

capture() saves frames from a side camera whenever a checkerboard is visible;
solve() computes K and D from saved frames and writes
calib/csi_<side>_intrinsics.yaml in the format StallLineDetector reads."""

import glob
import os
import time

import cv2
import numpy as np

from .config import CAMERA_CAPTURE_SIZE, CAMERA_INTRINSICS_FILE, PROC_SIZE, SCRIPT_DIR

MIN_IMAGES = 8                  # fewer usable images than this = refuse to calibrate
GOOD_RMS_PX = 0.5               # reprojection error below this is good
BAD_RMS_PX = 1.0                # above this, recapture
UPSCALE_BELOW_WIDTH = 640       # retry board detection at 2x on frames narrower than this


def frames_dir(side):
    return os.path.join(SCRIPT_DIR, "calib", "intrinsics_frames_" + side)


def find_corners(img, board):
    """board = (inner corners across, inner corners down). Returns Nx1x2 corners or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, board, flags=flags)
    if not ok and gray.shape[1] < UPSCALE_BELOW_WIDTH:
        # small frames (e.g. 420x220): squares can be only a few pixels wide; retry at 2x
        big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        ok, corners = cv2.findChessboardCorners(big, board, flags=flags)
        if ok:
            corners = (corners + 0.5) / 2.0 - 0.5        # back to original pixel coordinates
    if not ok:
        return None
    # refine with a search window smaller than the squares, or corners drift onto
    # neighbouring squares when the board is small in the image
    grid = corners.reshape(board[1], board[0], 2)
    spacing = min(np.linalg.norm(np.diff(grid, axis=1), axis=2).min(),
                  np.linalg.norm(np.diff(grid, axis=0), axis=2).min())
    win = int(max(1, min(5, spacing * 0.4)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    return cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), criteria)


def capture(side, board, num_frames, interval):
    """Save frames that contain the checkerboard. Numbering continues after any
    images already in the folder, so you can run it again to add more."""
    from .camera import SideCamera      # imported here so solve() works without PAL

    out_dir = frames_dir(side)
    os.makedirs(out_dir, exist_ok=True)
    index = len(glob.glob(os.path.join(out_dir, "*.png")))
    camera = SideCamera(side)
    print("Hold the checkerboard ({}x{} inner corners) in view of the {} camera.\n"
          "A frame is saved whenever the whole board is found; move it between saves:\n"
          "  center and all four edges/corners of the image, tilted different ways (not too steeply).\n"
          "  Keep it close: the board should fill about a third of the image or more.\n"
          "Saving {} frames to {} (Ctrl+C to stop early).".format(
              board[0], board[1], side, num_frames, out_dir))
    saved = 0
    next_t = time.monotonic()
    last_msg = 0.0
    try:
        while saved < num_frames:
            frame = camera.read()
            if frame is None or time.monotonic() < next_t:
                time.sleep(0.01)
                continue
            if saved == 0 and (frame.shape[1], frame.shape[0]) != tuple(CAMERA_CAPTURE_SIZE):
                print("[WARN] frames are {}x{} but CAMERA_CAPTURE_SIZE is {}x{}".format(
                    frame.shape[1], frame.shape[0], CAMERA_CAPTURE_SIZE[0], CAMERA_CAPTURE_SIZE[1]))
            if find_corners(frame, board) is None:
                if time.monotonic() - last_msg > 1.0:
                    print("  board not found - whole board in view, well lit, not blurred?")
                    last_msg = time.monotonic()
                continue
            path = os.path.join(out_dir, "{:03d}.png".format(index))
            cv2.imwrite(path, frame)
            index += 1
            saved += 1
            next_t = time.monotonic() + interval
            print("[{}/{}] board found, saved {} - move it to a new spot".format(saved, num_frames, path))
    except KeyboardInterrupt:
        print()
    finally:
        camera.terminate()
    return saved


def _coverage(img_pts, size, cells=4):
    """cells x cells grid over the image: True where any board corner was seen."""
    hit = np.zeros((cells, cells), bool)
    for pts in img_pts:
        p = pts.reshape(-1, 2)
        cx = np.clip((p[:, 0] / size[0] * cells).astype(int), 0, cells - 1)
        cy = np.clip((p[:, 1] / size[1] * cells).astype(int), 0, cells - 1)
        hit[cy, cx] = True
    return hit


def fisheye_valid_limits(K, D, size):
    """Where the fitted fisheye model stops being usable.

    The model maps the true angle off the optical axis, theta, to the image angle
    theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8). It is
    only usable while theta_d keeps increasing; with poor edge coverage the fit can
    turn back down, and everything in the image beyond that point is garbage after
    undistortion. Returns {edge name: fraction of the way to that edge that is valid}."""
    k = np.asarray(D, float).reshape(-1)[:4]
    th = np.linspace(0.0, np.pi, 3142)
    thd = th * (1 + k[0] * th**2 + k[1] * th**4 + k[2] * th**6 + k[3] * th**8)
    stop = np.where(np.diff(thd) <= 0)[0]
    thd_max = thd[stop[0]] if stop.size else thd[-1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w, h = size
    need = {"left edge": cx / fx, "right edge": (w - cx) / fx,
            "top edge": cy / fy, "bottom edge": (h - cy) / fy,
            "bottom corners": np.hypot(max(cx, w - cx) / fx, (h - cy) / fy)}
    return {name: min(1.0, thd_max / n) for name, n in need.items()}


def solve(side, board, square, model, images_dir=None, simple=False):
    """Calibrate from saved images and write the intrinsics file. Returns the RMS error [px].
    simple=True (fisheye only) fixes k3 = k4 = 0: a little less exact, but it can't
    fold back at the image edges when the board never reached them."""
    images_dir = images_dir or frames_dir(side)
    paths = sorted(p for p in glob.glob(os.path.join(images_dir, "*.png")) +
                   glob.glob(os.path.join(images_dir, "*.jpg"))
                   if not os.path.basename(p).startswith("preview"))
    objp = np.zeros((board[0] * board[1], 3), np.float64)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square

    obj_pts, img_pts, size = [], [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        s = (img.shape[1], img.shape[0])
        if size is None:
            size = s
        elif s != size:
            print("[SKIP] {}: size {}x{} differs from {}x{}".format(p, s[0], s[1], size[0], size[1]))
            continue
        corners = find_corners(img, board)
        if corners is None:
            print("[SKIP] {}: board not found".format(p))
            continue
        obj_pts.append(objp.copy())
        img_pts.append(corners.astype(np.float64))

    print("Using {} of {} images from {}".format(len(img_pts), len(paths), images_dir))
    if len(img_pts) < MIN_IMAGES:
        raise RuntimeError("need at least {} images with the board found (got {}) - "
                           "capture more".format(MIN_IMAGES, len(img_pts)))
    if size != tuple(CAMERA_CAPTURE_SIZE):
        print("[WARN] images are {}x{} but CAMERA_CAPTURE_SIZE is {}x{} - calibrate at the "
              "capture size you drive with".format(size[0], size[1], *CAMERA_CAPTURE_SIZE))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    try:
        if model == "fisheye":
            # numeric fallbacks: OpenCV 5 dropped these names from cv2.fisheye (values unchanged)
            flags = (getattr(cv2.fisheye, "CALIB_RECOMPUTE_EXTRINSIC", 2) |
                     getattr(cv2.fisheye, "CALIB_FIX_SKEW", 8))
            if simple:
                flags |= (getattr(cv2.fisheye, "CALIB_FIX_K3", 64) |
                          getattr(cv2.fisheye, "CALIB_FIX_K4", 128))
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                [o.reshape(-1, 1, 3) for o in obj_pts], [i.reshape(-1, 1, 2) for i in img_pts],
                size, np.zeros((3, 3)), np.zeros((4, 1)), flags=flags, criteria=criteria)
        else:
            rms, K, D, _, _ = cv2.calibrateCamera(
                [o.astype(np.float32) for o in obj_pts], [i.astype(np.float32) for i in img_pts],
                size, None, None, criteria=criteria)
    except cv2.error as e:
        raise RuntimeError("OpenCV could not calibrate ({}). Usually one bad image: remove "
                           "blurry or extreme-angle frames and retry.".format(str(e).strip()))

    out_path = CAMERA_INTRINSICS_FILE[side]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
    fs.write("model", model)
    fs.write("image_width", int(size[0]))
    fs.write("image_height", int(size[1]))
    fs.write("K", K)
    fs.write("D", D)
    fs.release()

    hit = _coverage(img_pts, size)
    quality = "good" if rms < GOOD_RMS_PX else ("OK" if rms < BAD_RMS_PX else "POOR - recapture")
    print("[INTRINSICS] model {}{} | RMS reprojection error {:.3f} px ({}) | image coverage {:.0f}%"
          .format(model, " (simple: k3 = k4 = 0)" if simple else "", rms, quality, 100 * hit.mean()))
    print("Coverage map (# = board seen there, . = never; top row = top of the image):")
    for row in hit:
        print("    " + " ".join("#" if c else "." for c in row))
    if hit.mean() < 0.75:
        print("[WARN] the board never reached the '.' areas - add frames with it there "
              "(the BOTTOM row matters most: that's where the floor is)")
    print("K =\n{}\nD = {}".format(np.array2string(K, precision=2), D.ravel()))
    if model == "fisheye":
        limits = fisheye_valid_limits(K, D, size)
        bad = {n: f for n, f in limits.items() if f < 1.0}
        if bad:
            print("[WARN] LENS MODEL FOLDS BACK before reaching: " +
                  ", ".join("{} (valid to {:.0f}%)".format(n, 100 * f) for n, f in bad.items()))
            print("       Undistorted images will be garbage beyond that. Fix: add frames near "
                  "those edges and corners and re-run, or re-run with --simple.")
        else:
            print("Lens model is valid all the way to the image edges and corners.")
    print("Wrote {}".format(out_path))
    _save_preview(side, paths[0], images_dir)
    return rms


def _save_preview(side, first_image, images_dir):
    """Undistort one image with the same loader/maps the detector uses, as a visual check."""
    from .stall_detector import Undistorter
    img = cv2.imread(first_image)
    small = cv2.resize(img, tuple(PROC_SIZE), interpolation=cv2.INTER_AREA)
    und = Undistorter(side).undistort(small)
    path = os.path.join(images_dir, "preview_undistorted.png")
    cv2.imwrite(path, np.hstack([small, und]))
    print("Saved {} (left: raw, right: undistorted - the board's edges should be straight)".format(path))
