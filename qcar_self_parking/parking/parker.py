"""Perpendicular parking state machine (GYRO_CAL -> SEARCH -> POSITION -> TURN -> ENTER -> DONE/ABORT)."""

import math
import signal
import time

import numpy as np

from pal.products.qcar import QCar

from .camera import SideCamera
from .config import (CAR_LENGTH, CENTER_FROM_REAR_AXLE, CAMERA_LATENCY_S, DT, DT_MAX,
                     ENTER_SPEED, ENTRANCE_MARGIN, GAP_DEPTH_JUMP, GYRO_CAL_DURATION,
                     GYRO_CAL_MAX_ATTEMPTS, GYRO_CAL_MAX_STD, GYRO_Z_SIGN, K_CENTER,
                     K_HEADING, LIDAR_FROM_REAR_AXLE, LIDAR_TO_FRONT_BUMPER,
                     LIDAR_TO_REAR_BUMPER, MAX_CORRECTION_STEER, MAX_GAP_WIDTH,
                     MAX_RUN_TIME, MAX_SEARCH_DISTANCE, MAX_STEER, MIN_GAP_WIDTH,
                     POSITION_SPEED, REAR_OVERHANG, ROW_CLEARANCE_MARGIN,
                     ROW_DETECT_MAX_DIST, SAFETY_LOG_PERIOD, SAFETY_SECTOR_HALF_WIDTH,
                     SAFETY_STOP_DIST, SAMPLE_RATE, SEARCH_SPEED,
                     STALL_CONFIRM_FRAMES, STALL_MATCH_TOL, STALL_MATCH_TOL_LATERAL,
                     STALL_MAX_MISSES, STALL_SEARCH_SPEED, STALL_WIDTH, STALL_WIDTH_TOL,
                     TIMING_LOG_PERIOD, TURN_ARC_TOLERANCE, TURN_RADIUS, TURN_SPEED)
from .control import SpeedController
from .lidar import LidarView
from .stall_detector import StallLineDetector
from .utils import leds


class PerpendicularParker:
    def __init__(self, mode, entry, side):
        self.mode = mode                      # "vehicles" | "empty"
        self.entry = entry                    # "reverse" | "forward"
        self.side = side
        self.k = 1 if side == "right" else -1 # +1 right, -1 left
        self.dir = -1 if entry == "reverse" else 1

        # load camera calibration before touching hardware, so a missing file fails cleanly
        self.detector = StallLineDetector(side) if mode == "empty" else None
        self.camera = None

        self.car = QCar(readMode=0, frequency=SAMPLE_RATE)
        self.lidar = LidarView()
        if mode == "empty":
            try:
                self.camera = SideCamera(side)
            except Exception:
                self.car.terminate()
                self.lidar.terminate()
                raise
        self.speed_ctrl = SpeedController()

        self.odom = 0.0        # [m] rear-axle path length (signed: + fwd, - rev)
        self.heading = 0.0     # [rad] relative to start, + = left
        self.speed = 0.0
        self.gyro_bias = 0.0   # [rad/s] raw gyroscope[2] at rest
        self._last_t = time.monotonic()
        self.abort_reason = None

        # Spot description in the start frame:
        #   spot_x      = along-aisle position of spot center (odom frame, rear axle)
        #   entry_depth = straight distance to travel after the 90 deg arc
        self.spot_x = None
        self.entry_depth = None
        self.turn_start_x = None

        # Vehicles-mode search
        self.baseline_side = None
        self.gap_start = None
        self.gap_row_dist = None   # distance from car centerline to neighbors' bumpers

        # Empty-mode stall tracking + diagnostics
        self.stall_track = []      # [(x_odom, lateral)] of consecutive matching detections
        self.stall_misses = 0
        self.cam_frames = 0
        self.frames_with_lines = 0
        self.frames_with_stall = 0
        self.frames_entrance_cut = 0
        self.best_spacing = None   # adjacent-line spacing closest to STALL_WIDTH seen so far
        self.best_run = 0

        self.turn_start_odom = None
        self.enter_start_odom = None
        # reverse-in turns the nose away from the spot, forward-in toward it
        self.target_heading = (self.k if entry == "reverse" else -self.k) * math.pi / 2.0

        self.state = "SEARCH"

        self._stop = False
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, *_):
        print("\n[!] Ctrl+C - stopping")
        self._stop = True

    def _abort(self, reason):
        self.abort_reason = reason
        self.state = "ABORT"
        print(f"[ABORT] {reason}")

    # -----------------------------------------------------------------
    # Planning
    # -----------------------------------------------------------------
    def _turn_start_from_center(self, spot_x):
        # reverse-in: rear axle moves backward R while turning -> start R past center
        # forward-in: rear axle moves forward R while turning -> start R before center
        return spot_x + TURN_RADIUS if self.entry == "reverse" else spot_x - TURN_RADIUS

    def _plan_empty(self, spot_x, spot_lateral):
        """spot_x: stall center along the aisle (odom frame, rear axle).
        spot_lateral: car centerline -> stall center, toward --side."""
        self.spot_x = spot_x
        # After the arc the rear axle is TURN_RADIUS deep. The car's geometric
        # center is CENTER_FROM_REAR_AXLE toward the aisle (reverse-in) or
        # deeper into the spot (forward-in).
        if self.entry == "reverse":
            center_depth_after_arc = TURN_RADIUS - CENTER_FROM_REAR_AXLE
        else:
            center_depth_after_arc = TURN_RADIUS + CENTER_FROM_REAR_AXLE
        self.entry_depth = spot_lateral - center_depth_after_arc
        if self.entry_depth < 0:
            print(f"[WARN] spot_lateral {spot_lateral:.2f} m is closer than the turn allows "
                  f"({center_depth_after_arc:.2f} m). Car will end {-self.entry_depth:.2f} m too deep. "
                  f"Start further from the spot.")
            self.entry_depth = 0.0
        self.turn_start_x = self._turn_start_from_center(self.spot_x)
        print(f"[PLAN] empty lot | R={TURN_RADIUS:.3f} m | spot center x={self.spot_x:.2f} m, "
              f"lateral={spot_lateral:.2f} m | turn at x={self.turn_start_x:.2f} m "
              f"(now {self.odom:.2f}) | enter {self.entry_depth:.2f} m after arc")

    def _leading_bumper_depth_after_arc(self):
        if self.entry == "reverse":
            return TURN_RADIUS + REAR_OVERHANG
        return TURN_RADIUS + (CAR_LENGTH - REAR_OVERHANG)

    def _plan_vehicles(self, gap_start, gap_end):
        """Returns True if a plan was made, False if the maneuver was refused."""
        width = gap_end - gap_start
        # gap edges are already in the odom frame (odom + LIDAR_FROM_REAR_AXLE),
        # so the midpoint is the spot center.
        self.spot_x = 0.5 * (gap_start + gap_end)
        row = self.gap_row_dist

        required = self._leading_bumper_depth_after_arc() + ROW_CLEARANCE_MARGIN
        if row < required:
            self._abort(
                f"row too close for a single-arc {self.entry}-in: row distance {row:.2f} m < "
                f"required {required:.2f} m (leading bumper depth after arc "
                f"{self._leading_bumper_depth_after_arc():.2f} m + margin {ROW_CLEARANCE_MARGIN:.2f} m). "
                f"Drive at least {required:.2f} m from the parked cars' bumpers and retry.")
            return False

        # Pull in until the bumper nearest the aisle is ENTRANCE_MARGIN past the neighbors.
        if self.entry == "reverse":
            # nose faces aisle: front bumper is (CAR_LENGTH - REAR_OVERHANG) ahead of rear axle
            rear_axle_depth = row + ENTRANCE_MARGIN + (CAR_LENGTH - REAR_OVERHANG)
        else:
            # rear faces aisle: rear bumper is REAR_OVERHANG behind rear axle
            rear_axle_depth = row + ENTRANCE_MARGIN + REAR_OVERHANG
        self.entry_depth = max(0.0, rear_axle_depth - TURN_RADIUS)
        self.turn_start_x = self._turn_start_from_center(self.spot_x)
        print(f"[PLAN] gap width {width:.2f} m, row distance {row:.2f} m | "
              f"spot center x={self.spot_x:.2f} m | turn at x={self.turn_start_x:.2f} m "
              f"(now {self.odom:.2f}) | enter {self.entry_depth:.2f} m after arc")
        return True

    # -----------------------------------------------------------------
    # Low level
    # -----------------------------------------------------------------
    def _tick(self):
        """Measured, clamped time since the last integration step."""
        now = time.monotonic()
        dt = min(max(now - self._last_t, 0.0), DT_MAX)
        self._last_t = now
        return dt

    def _integrate(self, dt):
        # ASSUMPTION: motorTach is linear car speed in m/s (see header notes)
        self.speed = float(self.car.motorTach)
        self.odom += self.speed * dt
        gz = float(self.car.gyroscope[2]) - self.gyro_bias
        self.heading += GYRO_Z_SIGN * gz * dt

    def drive(self, target_speed, steering, led_arr):
        dt = self._tick()
        throttle = self.speed_ctrl.step(target_speed, self.speed, dt)
        self.car.read_write_std(throttle=throttle,
                                steering=float(np.clip(steering, -MAX_STEER, MAX_STEER)),
                                LEDs=led_arr)
        self._integrate(dt)

    def stop_car(self, led_arr=None, settle=0.0):
        """Zero throttle for `settle` seconds, still integrating odom/heading
        so any coast-down is counted."""
        self.speed_ctrl.reset()
        if led_arr is None:
            led_arr = leds(brake=True)
        t_end = time.monotonic() + max(settle, 5 * DT)
        while time.monotonic() < t_end:
            dt = self._tick()
            self.car.read_write_std(throttle=0.0, steering=0.0, LEDs=led_arr)
            self._integrate(dt)
            time.sleep(DT)

    def calibrate_gyro(self):
        """Average gyroscope[2] with the car stationary. Returns True on success."""
        for attempt in range(1, GYRO_CAL_MAX_ATTEMPTS + 1):
            print(f"[GYRO_CAL] hold the car still ({GYRO_CAL_DURATION:.0f} s, attempt {attempt})...")
            samples = []
            t_end = time.monotonic() + GYRO_CAL_DURATION
            while time.monotonic() < t_end and not self._stop:
                self.car.read_write_std(throttle=0.0, steering=0.0, LEDs=leds(brake=True))
                samples.append(float(self.car.gyroscope[2]))
                time.sleep(DT)
            if self._stop:
                return False
            if len(samples) < 10:
                print(f"[GYRO_CAL] only {len(samples)} samples - retrying")
                continue
            mean, std = float(np.mean(samples)), float(np.std(samples))
            if std > GYRO_CAL_MAX_STD:
                print(f"[GYRO_CAL][WARN] noisy: std {std:.4f} rad/s > {GYRO_CAL_MAX_STD:.4f} "
                      f"(car bumped or moving?) - re-running calibration")
                continue
            self.gyro_bias = mean
            print(f"[GYRO_CAL] bias {mean:+.5f} rad/s, std {std:.5f} rad/s ({len(samples)} samples)")
            return True
        self._abort(f"gyro calibration failed {GYRO_CAL_MAX_ATTEMPTS} times "
                    f"(too noisy) - keep the car still and retry")
        return False

    def obstacle_ahead(self, target_speed):
        """True if something is too close in the direction of travel. The closest
        reading is kept in self.safety_reading for the log messages."""
        if target_speed > 0:
            which, center, limit = "front", 0.0, LIDAR_TO_FRONT_BUMPER + SAFETY_STOP_DIST
        elif target_speed < 0:
            which, center, limit = "rear", 180.0, LIDAR_TO_REAR_BUMPER + SAFETY_STOP_DIST
        else:
            return False
        dist, angle = self.lidar.nearest(center, SAFETY_SECTOR_HALF_WIDTH)
        self.safety_reading = (which, dist, angle, limit)
        return dist < limit

    def _safety_text(self):
        which, dist, angle, limit = self.safety_reading
        return f"{which} {dist:.2f} m at {angle:+.0f} deg (limit {limit:.2f} m)"

    def indicator(self, brake=False):
        return leds(right=self.k == 1, left=self.k == -1, brake=brake)

    # -----------------------------------------------------------------
    # States
    # -----------------------------------------------------------------
    def step_search_vehicles(self):
        side = self.lidar.side(self.k)
        lidar_x = self.odom + LIDAR_FROM_REAR_AXLE
        steer = float(np.clip(K_HEADING * (0.0 - self.heading),
                              -MAX_CORRECTION_STEER, MAX_CORRECTION_STEER))

        if self.odom >= MAX_SEARCH_DISTANCE:
            if self.baseline_side is None:
                self._abort(f"side beam never saw a parked car (closer than "
                            f"{ROW_DETECT_MAX_DIST:.2f} m) within {MAX_SEARCH_DISTANCE:.1f} m")
            elif self.gap_start is not None:
                self._abort(f"no usable gap within {MAX_SEARCH_DISTANCE:.1f} m "
                            f"(a gap was still open, {lidar_x - self.gap_start:.2f} m so far)")
            else:
                self._abort(f"no gap >= {MIN_GAP_WIDTH:.2f} m found within {MAX_SEARCH_DISTANCE:.1f} m")
            return 0.0, 0.0, leds(brake=True)

        if self.baseline_side is None:
            if np.isfinite(side) and side < ROW_DETECT_MAX_DIST:
                self.baseline_side = side
                print(f"[SEARCH] row distance = {side:.2f} m")
            return SEARCH_SPEED, steer, self.indicator()

        is_gap = (not np.isfinite(side)) or (side > self.baseline_side + GAP_DEPTH_JUMP)

        if self.gap_start is None:
            if is_gap:
                self.gap_start = lidar_x
                self.gap_row_dist = self.baseline_side
                print(f"[SEARCH] gap opens at x={self.gap_start:.2f} m")
            else:
                self.baseline_side = 0.9 * self.baseline_side + 0.1 * side
        else:
            width = lidar_x - self.gap_start
            if not is_gap:
                if width >= MIN_GAP_WIDTH:
                    print(f"[SEARCH] gap closes at x={lidar_x:.2f} m, width {width:.2f} m - OK")
                    if self._plan_vehicles(self.gap_start, lidar_x):
                        self.state = "POSITION"
                    else:
                        return 0.0, 0.0, leds(brake=True)
                else:
                    print(f"[SEARCH] gap too narrow ({width:.2f} < {MIN_GAP_WIDTH:.2f} m) - keep looking")
                    self.gap_start = None
                    self.baseline_side = side
            elif width >= MAX_GAP_WIDTH:
                # open space after the last car: use a spot right next to it
                print(f"[SEARCH] open space after x={self.gap_start:.2f} m - using spot beside last vehicle")
                if self._plan_vehicles(self.gap_start, self.gap_start + MIN_GAP_WIDTH + 0.05):
                    self.state = "POSITION"
                else:
                    return 0.0, 0.0, leds(brake=True)

        return SEARCH_SPEED, steer, self.indicator()

    def _stall_search_failure_reason(self):
        d = f"within {MAX_SEARCH_DISTANCE:.1f} m"
        if self.cam_frames == 0:
            return f"no frames received from the {self.side} CSI camera {d}"
        if self.frames_with_lines == 0:
            return (f"no perpendicular tape lines seen in {self.cam_frames} frames {d} "
                    f"(check TAPE_HSV_LOW/HIGH and the BEV with --calibrate-camera)")
        if self.frames_with_stall == 0:
            best = "none" if self.best_spacing is None else f"{self.best_spacing:.2f} m"
            extra = (f"; {self.frames_entrance_cut} frames had a matching pair whose entrance end "
                     f"was outside the camera view") if self.frames_entrance_cut else ""
            return (f"lines seen in {self.frames_with_lines}/{self.cam_frames} frames {d}, but no "
                    f"adjacent pair matched STALL_WIDTH {STALL_WIDTH:.2f} +/- {STALL_WIDTH_TOL:.2f} m "
                    f"(closest spacing seen: {best}){extra}")
        return (f"stall seen in {self.frames_with_stall} frames {d} but never confirmed in "
                f"{STALL_CONFIRM_FRAMES} consecutive frames (best run {self.best_run})")

    def step_search_stall(self):
        steer = float(np.clip(K_HEADING * (0.0 - self.heading),
                              -MAX_CORRECTION_STEER, MAX_CORRECTION_STEER))

        if self.odom >= MAX_SEARCH_DISTANCE:
            self._abort(self._stall_search_failure_reason())
            return 0.0, 0.0, leds(brake=True)

        frame = self.camera.read()
        if frame is None:
            return STALL_SEARCH_SPEED, steer, self.indicator()   # no new frame this loop

        odom_at_frame = self.odom - self.speed * CAMERA_LATENCY_S
        psi = self.heading
        det = self.detector.detect(frame)
        self.cam_frames += 1
        if det["clusters"]:
            self.frames_with_lines += 1
        if det["rejected_entrance"]:
            self.frames_entrance_cut += 1
        for s in det["spacings"]:
            if self.best_spacing is None or abs(s - STALL_WIDTH) < abs(self.best_spacing - STALL_WIDTH):
                self.best_spacing = s

        # car frame -> start frame (odom along the aisle, lateral toward --side).
        # Assumes the car's lateral drift from the start line is ~0 (heading hold).
        cands = []
        for st in det["stalls"]:
            f, out = st["fwd"], st["lateral"]
            x = odom_at_frame + f * math.cos(psi) + self.k * out * math.sin(psi)
            lat = out * math.cos(psi) - self.k * f * math.sin(psi)
            cands.append((x, lat))
        if cands:
            self.frames_with_stall += 1

        match = None
        if self.stall_track:
            mx = float(np.mean([t[0] for t in self.stall_track]))
            ml = float(np.mean([t[1] for t in self.stall_track]))
            close = [c for c in cands
                     if abs(c[0] - mx) <= STALL_MATCH_TOL and abs(c[1] - ml) <= STALL_MATCH_TOL_LATERAL]
            if close:
                match = min(close, key=lambda c: abs(c[0] - mx))
            else:
                self.stall_misses += 1
                if self.stall_misses > STALL_MAX_MISSES:
                    self.stall_track = []
                    self.stall_misses = 0
        if match is None and not self.stall_track and cands:
            match = min(cands, key=lambda c: c[0])      # start tracking the nearest-along-aisle stall

        if match is not None:
            self.stall_track.append(match)
            self.stall_misses = 0
            self.best_run = max(self.best_run, len(self.stall_track))
            print(f"[SEARCH] stall candidate x={match[0]:.2f} m, lateral={match[1]:.2f} m "
                  f"({len(self.stall_track)}/{STALL_CONFIRM_FRAMES})")

        if len(self.stall_track) >= STALL_CONFIRM_FRAMES:
            xs = np.array([t[0] for t in self.stall_track])
            ls = np.array([t[1] for t in self.stall_track])
            print(f"[SEARCH] stall confirmed: x={xs.mean():.3f} m (std {xs.std():.3f}), "
                  f"lateral={ls.mean():.3f} m (std {ls.std():.3f}) over {len(xs)} frames")
            self._plan_empty(float(xs.mean()), float(ls.mean()))
            self.state = "POSITION"

        return STALL_SEARCH_SPEED, steer, self.indicator()

    def step_position(self):
        err = self.turn_start_x - self.odom
        if abs(err) < 0.01:
            self.stop_car(self.indicator(brake=True), settle=0.5)
            self.turn_start_odom = self.odom
            self.state = "TURN"
            print(f"[POSITION] at turn point (heading {math.degrees(self.heading):.1f} deg) - turning")
            return 0.0, 0.0, self.indicator(brake=True)
        speed = math.copysign(POSITION_SPEED if abs(err) > 0.10 else 0.06, err)
        steer = K_HEADING * (0.0 - self.heading) * (1 if speed > 0 else -1)
        steer = float(np.clip(steer, -MAX_CORRECTION_STEER, MAX_CORRECTION_STEER))
        return speed, steer, self.indicator()

    def step_turn(self):
        travelled = abs(self.odom - self.turn_start_odom)
        expected_arc = TURN_RADIUS * math.pi / 2.0
        reached = abs(self.heading) >= abs(self.target_heading) - math.radians(2.0)
        if reached or travelled > TURN_ARC_TOLERANCE * expected_arc:
            if not reached:
                print(f"[WARN] arc length exceeded, heading only {math.degrees(self.heading):.1f} deg")
            self.stop_car(self.indicator(brake=True), settle=0.3)
            self.enter_start_odom = self.odom
            self.state = "ENTER"
            print(f"[TURN] done, heading {math.degrees(self.heading):.1f} deg - entering spot")
            return 0.0, 0.0, self.indicator(brake=True)
        # steer toward the spot side for both entry directions
        return self.dir * TURN_SPEED, -self.k * MAX_STEER, self.indicator()

    def step_enter(self):
        travelled = abs(self.odom - self.enter_start_odom)
        if travelled >= self.entry_depth:
            self.state = "DONE"
            return 0.0, 0.0, leds(brake=True)

        err_h = self.target_heading - self.heading
        steer = self.dir * K_HEADING * err_h

        if self.mode == "vehicles":
            left = self.lidar.sector(90.0, 10.0)
            right = self.lidar.sector(-90.0, 10.0)
            if np.isfinite(left) and np.isfinite(right) and left < 0.5 and right < 0.5:
                # positive -> car is too far right -> move left (+ steer works for both directions)
                lateral_err = 0.5 * (left - right)
                steer += K_CENTER * lateral_err

        steer = float(np.clip(steer, -MAX_CORRECTION_STEER, MAX_CORRECTION_STEER))
        return self.dir * ENTER_SPEED, steer, leds()

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    def run(self):
        print(f"Perpendicular parking | mode={self.mode} entry={self.entry} "
              f"side={self.side} | Ctrl+C to abort")
        steps = {
            "SEARCH": self.step_search_vehicles if self.mode == "vehicles" else self.step_search_stall,
            "POSITION": self.step_position,
            "TURN": self.step_turn,
            "ENTER": self.step_enter,
        }
        try:
            if not self.calibrate_gyro():
                return
            self.heading = 0.0
            self.odom = 0.0
            self._last_t = time.monotonic()

            t0 = time.monotonic()
            prev_loop = t0
            log_t = t0
            loop_times = []
            safety_log_t = t0 - SAFETY_LOG_PERIOD
            safety_blocked = 0

            while not self._stop and self.state not in ("DONE", "ABORT"):
                loop_start = time.monotonic()
                loop_times.append(loop_start - prev_loop)
                prev_loop = loop_start
                if loop_start - log_t >= TIMING_LOG_PERIOD and loop_times:
                    print(f"[TIMING] loop avg {1000 * np.mean(loop_times):.1f} ms, "
                          f"max {1000 * np.max(loop_times):.1f} ms "
                          f"({len(loop_times)} iterations) | state {self.state} | "
                          f"odom {self.odom:+.3f} m, heading {math.degrees(self.heading):+.1f} deg | "
                          f"{self.side} side {self.lidar.side(self.k):.2f} m"
                          + ("" if self.baseline_side is None else
                             f" (row {self.baseline_side:.2f} m, gap if > "
                             f"{self.baseline_side + GAP_DEPTH_JUMP:.2f})"))
                    loop_times = []
                    log_t = loop_start

                if loop_start - t0 > MAX_RUN_TIME:
                    print("[!] timeout")
                    break

                self.lidar.update()
                target_speed, steer, led_arr = steps[self.state]()

                if self.obstacle_ahead(target_speed):
                    if self.state == "ENTER":
                        # curb / wall at the back (or front) of the spot - we're in
                        print(f"[ENTER] reached end of spot: {self._safety_text()}")
                        self.state = "DONE"
                    else:
                        # one message per SAFETY_LOG_PERIOD instead of one per loop
                        safety_blocked += 1
                        if loop_start - safety_log_t >= SAFETY_LOG_PERIOD:
                            print(f"[SAFETY] holding during {self.state}: {self._safety_text()} "
                                  f"| blocked {safety_blocked} loops")
                            safety_blocked = 0
                            safety_log_t = loop_start
                    target_speed = 0.0
                    led_arr = leds(brake=True)

                self.drive(target_speed, steer, led_arr)

                elapsed = time.monotonic() - loop_start
                if elapsed < DT:
                    time.sleep(DT - elapsed)

            if self.state == "DONE":
                self.lidar.update()
                print(f"[DONE] parked. heading error "
                      f"{math.degrees(self.heading - self.target_heading):+.1f} deg | "
                      f"left {self.lidar.sector(90, 10):.2f} m, right {self.lidar.sector(-90, 10):.2f} m")
            elif self.state == "ABORT":
                print(f"[ABORT] stopped without parking: {self.abort_reason}")
        finally:
            self.stop_car(leds(left=True, right=True, brake=True))   # hazards
            self.car.terminate()
            self.lidar.terminate()
            if self.camera is not None:
                self.camera.terminate()
