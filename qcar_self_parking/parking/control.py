"""Speed PI controller."""

import numpy as np

from .config import KP_SPEED, KI_SPEED, MAX_THROTTLE


class SpeedController:
    def __init__(self):
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def step(self, target, measured, dt):
        if target == 0.0:
            self.reset()
            return 0.0
        err = target - measured
        self.integral = float(np.clip(self.integral + err * dt, -0.3, 0.3))
        u = KP_SPEED * err + KI_SPEED * self.integral
        u += 0.04 * np.sign(target)          # feed-forward to overcome stiction
        return float(np.clip(u, -MAX_THROTTLE, MAX_THROTTLE))
