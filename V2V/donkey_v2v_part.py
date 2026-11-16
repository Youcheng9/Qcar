"""Donkey Car part that adds V2V to manage.py.

Sits between DriveMode and the motors: takes the car's planned throttle,
lowers it based on what the other car says, and tells the other car what we
are doing. It needs no donkeycar imports, so it can be tested on any computer.

Add to manage.py right after the DriveMode V.add(...) block:

    if getattr(cfg, 'V2V_ENABLED', False):
        import sys
        sys.path.insert(0, cfg.V2V_PATH)
        from donkey_v2v_part import V2VPart
        V.add(V2VPart(cfg.V2V_CAR_ID, cfg.V2V_PEER_IP),
              inputs=['user/mode', 'steering', 'throttle'],
              outputs=['steering', 'throttle'])

and in myconfig.py:

    V2V_ENABLED = True
    V2V_PATH = '/home/mydonkey/projects/Qcar/V2V'
    V2V_CAR_ID = 'donkey1'
    V2V_PEER_IP = '192.168.1.20'   # the other car, or None to broadcast
"""
import time

from v2v import V2VLink, limit_throttle

BRAKE_HOLD = 1.0       # keep reporting "braking" this long after stopping
TURN_THRESHOLD = 0.3   # donkeycar steering is -1 (left) .. +1 (right)


class V2VPart:
    def __init__(self, car_id, peer_ip=None, slow_factor=0.5):
        self.link = V2VLink(car_id, peer_ip=peer_ip).start()
        self.slow_factor = slow_factor
        self.last_moving = float("-inf")
        self.last_reason = None

    def run(self, mode, steering, throttle):
        steering = float(steering or 0.0)
        throttle = float(throttle or 0.0)
        now = time.time()

        # Our own decision. It is made BEFORE the V2V cap, so a car that was
        # only told to stop never reports "braking" itself. Otherwise two
        # cars would keep each other stopped forever.
        if throttle > 0.0:
            self.last_moving = now
        braking = throttle <= 0.0 and (now - self.last_moving) < BRAKE_HOLD

        # Only cap forward driving. Never touch reverse.
        capped = throttle
        if throttle > 0.0:
            peer = self.link.get_peer()
            capped, reason = limit_throttle(throttle, peer, self.slow_factor)
            if reason != self.last_reason:
                print("V2V:", reason, "| peer:", peer)
                self.last_reason = reason

        if steering > TURN_THRESHOLD:
            signal = "right"
        elif steering < -TURN_THRESHOLD:
            signal = "left"
        else:
            signal = "none"

        self.link.update_state(
            throttle=throttle, steering=steering, mode=str(mode),
            brake=braking, turn_signal=signal)
        return steering, capped

    def shutdown(self):
        self.link.update_state(throttle=0.0, brake=True)
        time.sleep(0.2)  # let one last "braking" packet go out
        self.link.stop()
