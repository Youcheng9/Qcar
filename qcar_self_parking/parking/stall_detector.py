"""Stall-line detection from the side camera (empty mode):
undistort -> bird's-eye view -> tape threshold -> HoughLinesP -> stall pairs."""

import math
import os

import cv2
import numpy as np

from .config import (CAMERA_BEV_FILE, CAMERA_CAPTURE_SIZE, CAMERA_COLOR_ORDER,
                     CAMERA_INTRINSICS_FILE, CAMERA_MOUNT, ENTRANCE_ALIGN_TOL,
                     ENTRANCE_EDGE_MARGIN_PX, FISHEYE_BALANCE,
                     FISHEYE_FALLBACK_FOCAL_SCALE, HOUGH_MAX_LINE_GAP,
                     HOUGH_MIN_LINE_LENGTH, HOUGH_THRESHOLD, LINE_ANGLE_TOL_DEG,
                     LINE_MERGE_DIST, MIN_LINE_EXTENT, MORPH_KERNEL_PX, PINHOLE_ALPHA,
                     PROC_SIZE, STALL_DEPTH, STALL_WIDTH, STALL_WIDTH_TOL,
                     TAPE_HSV_HIGH, TAPE_HSV_LOW)


def _open_fs(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{what} file not found: {path}\n"
            f"Create it first: intrinsics with calibrate_intrinsics.py, then the bird's-eye "
            f"homography with make_bev.py (see the top of qcar_self_parking.py).")
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"could not open {what} file: {path}")
    return fs


def _sane_camera_matrix(K, size):
    w, h = size
    return (np.all(np.isfinite(K)) and K[0, 0] > 0 and K[1, 1] > 0
            and 0 <= K[0, 2] <= w and 0 <= K[1, 2] <= h)


class Undistorter:
    """Loads one side camera's intrinsics and undistorts frames that were
    already downscaled to PROC_SIZE. Needs only the intrinsics file, so
    make_bev.py can use it before a bird's-eye homography exists."""

    def __init__(self, side):
        self._load_intrinsics(CAMERA_INTRINSICS_FILE[side])
        self._build_maps()

    def _load_intrinsics(self, path):
        fs = _open_fs(path, "camera intrinsics")
        self.model = (fs.getNode("model").string() or "pinhole").strip().lower()
        w = int(fs.getNode("image_width").real())
        h = int(fs.getNode("image_height").real())
        K = fs.getNode("K").mat()
        D = fs.getNode("D").mat()
        fs.release()
        if K is None or D is None or w <= 0 or h <= 0:
            raise ValueError(f"{path}: needs model, image_width, image_height, K, D")
        if (w, h) != tuple(CAMERA_CAPTURE_SIZE):
            print(f"[CAM][WARN] intrinsics were calibrated at {w}x{h} but capture is "
                  f"{CAMERA_CAPTURE_SIZE[0]}x{CAMERA_CAPTURE_SIZE[1]} - K is scaled, "
                  f"but only correct if the sensor crop is the same")
        K = K.astype(np.float64).copy()
        K[0, :] *= PROC_SIZE[0] / float(w)     # fx, skew, cx
        K[1, :] *= PROC_SIZE[1] / float(h)     # fy, cy
        self.K = K
        self.D = D.astype(np.float64)

    def _build_maps(self):
        size = tuple(PROC_SIZE)
        if self.model == "fisheye":
            D4 = self.D.reshape(-1)[:4].reshape(4, 1)
            K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                self.K, D4, size, np.eye(3), balance=FISHEYE_BALANCE)
            if not _sane_camera_matrix(K_new, size):
                # Happens with very wide lenses (image edges near/past 90 deg off-axis):
                # OpenCV's estimate blows up. Use the calibrated K, zoomed out instead.
                print("[CAM][WARN] OpenCV's fisheye undistortion estimate is unusable for this "
                      "wide lens - using calibrated focal length x {:.2f} "
                      "(FISHEYE_FALLBACK_FOCAL_SCALE)".format(FISHEYE_FALLBACK_FOCAL_SCALE))
                K_new = self.K.copy()
                K_new[0, 0] *= FISHEYE_FALLBACK_FOCAL_SCALE
                K_new[1, 1] *= FISHEYE_FALLBACK_FOCAL_SCALE
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, D4, np.eye(3), K_new, size, cv2.CV_32FC1)
        else:
            K_new, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, size, PINHOLE_ALPHA, size)
            m1, m2 = cv2.initUndistortRectifyMap(self.K, self.D, None, K_new, size, cv2.CV_32FC1)
        self.K_new = K_new
        self.map1, self.map2 = m1, m2

    def undistort(self, small):
        return cv2.remap(small, self.map1, self.map2, cv2.INTER_LINEAR)


class StallLineDetector:
    """Undistort + bird's-eye warp (one precomputed remap), tape threshold,
    HoughLinesP, perpendicular-line clustering, and stall-pair matching.
    All outputs are in the CAR frame: fwd = meters ahead of the rear axle,
    out = meters from the car centerline toward the camera's side."""

    def __init__(self, side):
        self.side = side
        self.mount_fwd, self.mount_out = CAMERA_MOUNT[side]
        self.undistorter = Undistorter(side)
        self._load_bev(CAMERA_BEV_FILE[side])
        self._build_maps()
        self.kernel = np.ones((MORPH_KERNEL_PX, MORPH_KERNEL_PX), np.uint8)
        self.hsv_code = cv2.COLOR_RGB2HSV if CAMERA_COLOR_ORDER == "RGB" else cv2.COLOR_BGR2HSV

    # ---- calibration loading / map precomputation ----
    def _load_bev(self, path):
        fs = _open_fs(path, "bird's-eye homography")
        H = fs.getNode("H").mat()
        proc = (int(fs.getNode("proc_width").real()), int(fs.getNode("proc_height").real()))
        self.bev_w = int(fs.getNode("bev_width").real())
        self.bev_h = int(fs.getNode("bev_height").real())
        self.mpp = float(fs.getNode("meters_per_px").real())
        self.origin_fwd = float(fs.getNode("origin_fwd_m").real())
        self.origin_out = float(fs.getNode("origin_out_m").real())
        fs.release()
        if H is None or self.bev_w <= 0 or self.bev_h <= 0 or self.mpp <= 0:
            raise ValueError(f"{path}: needs H, proc_width/height, bev_width/height, "
                             f"meters_per_px, origin_fwd_m, origin_out_m")
        if proc != tuple(PROC_SIZE):
            raise ValueError(f"{path}: homography was made at {proc[0]}x{proc[1]} but "
                             f"PROC_SIZE is {PROC_SIZE[0]}x{PROC_SIZE[1]} - redo it or change PROC_SIZE")
        self.H = H.astype(np.float64)

    def _build_maps(self):
        size = tuple(PROC_SIZE)
        m1, m2 = self.undistorter.map1, self.undistorter.map2

        # Compose undistort + homography into ONE remap from the downscaled raw
        # frame straight to the BEV, so each frame costs a single cv2.remap and
        # only floor pixels inside the BEV region are ever touched.
        cols, rows = np.meshgrid(np.arange(self.bev_w, dtype=np.float32),
                                 np.arange(self.bev_h, dtype=np.float32))
        pts = np.stack([cols, rows], axis=-1).reshape(-1, 1, 2)
        und = cv2.perspectiveTransform(pts, np.linalg.inv(self.H)).reshape(self.bev_h, self.bev_w, 2)
        ux = np.ascontiguousarray(und[..., 0], dtype=np.float32)
        uy = np.ascontiguousarray(und[..., 1], dtype=np.float32)
        self.bev_map_x = cv2.remap(m1, ux, uy, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        self.bev_map_y = cv2.remap(m2, ux, uy, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        w, h = size
        valid = ((self.bev_map_x >= 0) & (self.bev_map_x <= w - 1) &
                 (self.bev_map_y >= 0) & (self.bev_map_y <= h - 1))
        self.bev_valid = valid.astype(np.uint8) * 255
        # nearest (smallest) valid row per column = near edge of what the camera sees
        any_valid = valid.any(axis=0)
        self.near_valid_row = np.where(any_valid, valid.argmax(axis=0), self.bev_h)

    # ---- coordinate helpers ----
    def _px(self, meters):
        return max(1, int(round(meters / self.mpp)))

    def px_to_car(self, col, row):
        fwd = self.mount_fwd + self.origin_fwd + col * self.mpp
        out = self.mount_out + self.origin_out + row * self.mpp
        return fwd, out

    def car_to_px(self, fwd, out):
        col = (fwd - self.mount_fwd - self.origin_fwd) / self.mpp
        row = (out - self.mount_out - self.origin_out) / self.mpp
        return int(round(col)), int(round(row))

    def undistort(self, small):
        return self.undistorter.undistort(small)

    # ---- detection ----
    def detect(self, frame):
        small = cv2.resize(frame, tuple(PROC_SIZE), interpolation=cv2.INTER_AREA)
        bev = cv2.remap(small, self.bev_map_x, self.bev_map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        hsv = cv2.cvtColor(bev, self.hsv_code)
        mask = cv2.inRange(hsv, np.array(TAPE_HSV_LOW, np.uint8), np.array(TAPE_HSV_HIGH, np.uint8))
        mask = cv2.bitwise_and(mask, self.bev_valid)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        raw = cv2.HoughLinesP(mask, 1, np.pi / 180.0, HOUGH_THRESHOLD,
                              minLineLength=self._px(HOUGH_MIN_LINE_LENGTH),
                              maxLineGap=self._px(HOUGH_MAX_LINE_GAP))
        segs = [] if raw is None else [tuple(int(v) for v in s) for s in raw.reshape(-1, 4)]

        # keep segments roughly perpendicular to travel (= vertical in the BEV)
        kept = []
        for x1, y1, x2, y2 in segs:
            ang = math.degrees(math.atan2(abs(x2 - x1), abs(y2 - y1)))   # 0 = perpendicular
            if ang <= LINE_ANGLE_TOL_DEG:
                kept.append((x1, y1, x2, y2))

        clusters = self._cluster(kept)
        stalls, spacings, rejected_entrance = self._pair(clusters)
        return {"small": small, "bev": bev, "mask": mask, "segs": segs, "kept": kept,
                "clusters": clusters, "stalls": stalls, "spacings": spacings,
                "rejected_entrance": rejected_entrance}

    def _cluster(self, kept):
        items = []
        for x1, y1, x2, y2 in kept:
            f1, o1 = self.px_to_car(x1, y1)
            f2, o2 = self.px_to_car(x2, y2)
            items.append({"fwd": 0.5 * (f1 + f2), "len": math.hypot(f2 - f1, o2 - o1),
                          "near": min(o1, o2), "far": max(o1, o2),
                          "near_row": min(y1, y2), "far_row": max(y1, y2),
                          "col": 0.5 * (x1 + x2)})
        items.sort(key=lambda it: it["fwd"])

        groups = []
        for it in items:
            if groups:
                g = groups[-1]
                g_fwd = sum(i["fwd"] * i["len"] for i in g) / sum(i["len"] for i in g)
                if it["fwd"] - g_fwd <= LINE_MERGE_DIST:
                    g.append(it)
                    continue
            groups.append([it])

        clusters = []
        for g in groups:
            total = sum(i["len"] for i in g)
            near = min(i["near"] for i in g)
            far = max(i["far"] for i in g)
            if far - near < MIN_LINE_EXTENT:
                continue
            col = int(round(sum(i["col"] * i["len"] for i in g) / total))
            col = int(np.clip(col, 0, self.bev_w - 1))
            near_row = min(i["near_row"] for i in g)
            clusters.append({
                "fwd": sum(i["fwd"] * i["len"] for i in g) / total,
                "near": near, "far": far,
                "col": col, "near_row": near_row, "far_row": max(i["far_row"] for i in g),
                "entrance_visible": near_row > self.near_valid_row[col] + ENTRANCE_EDGE_MARGIN_PX,
            })
        return clusters

    def _pair(self, clusters):
        stalls, spacings, rejected_entrance = [], [], 0
        for a, b in zip(clusters, clusters[1:]):
            s = b["fwd"] - a["fwd"]
            spacings.append(s)
            if abs(s - STALL_WIDTH) > STALL_WIDTH_TOL:
                continue
            if not (a["entrance_visible"] and b["entrance_visible"]):
                rejected_entrance += 1
                continue
            if abs(a["near"] - b["near"]) > ENTRANCE_ALIGN_TOL:
                continue
            entrance = 0.5 * (a["near"] + b["near"])
            stalls.append({"fwd": 0.5 * (a["fwd"] + b["fwd"]),
                           "lateral": entrance + STALL_DEPTH / 2.0,
                           "spacing": s})
        return stalls, spacings, rejected_entrance

    # ---- debug drawing (for --calibrate-camera) ----
    def draw(self, det):
        vis = det["bev"].copy()
        invalid = det["bev"].copy()
        invalid[:] = (60, 0, 60)
        vis = np.where(self.bev_valid[..., None] > 0, vis, invalid)
        for x1, y1, x2, y2 in det["segs"]:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)          # red: all Hough segments
        for x1, y1, x2, y2 in det["kept"]:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)          # green: perpendicular
        for c in det["clusters"]:
            cv2.line(vis, (c["col"], c["near_row"]), (c["col"], c["far_row"]), (255, 0, 0), 2)
            color = (255, 255, 0) if c["entrance_visible"] else (0, 0, 255)
            cv2.circle(vis, (c["col"], c["near_row"]), 3, color, -1)   # cyan = entrance end seen
        for st in det["stalls"]:
            col, row = self.car_to_px(st["fwd"], st["lateral"])
            if 0 <= col < self.bev_w and 0 <= row < self.bev_h:
                cv2.circle(vis, (col, row), 5, (0, 255, 255), 2)       # yellow: stall center
            cv2.putText(vis, f"{st['spacing']:.2f}m", (max(0, col - 20), 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        bar = self._px(STALL_WIDTH)
        cv2.line(vis, (5, self.bev_h - 6), (5 + bar, self.bev_h - 6), (255, 255, 255), 2)
        cv2.putText(vis, "STALL_WIDTH", (5, self.bev_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return vis
