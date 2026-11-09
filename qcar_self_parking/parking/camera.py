"""Side CSI camera wrapper (empty mode)."""

import numpy as np

from pal.utilities.vision import Camera2D

from .config import CAMERA_CAPTURE_SIZE, CAMERA_FPS, CAMERA_ID


class SideCamera:
    """One side CSI camera via PAL Camera2D. read() returns a new frame or None."""

    def __init__(self, side, dedupe=True):
        # Same constructor as the Qcar-master programs that ran on a QCar 1
        self.cam = Camera2D(cameraId=CAMERA_ID[side],
                            frameWidth=CAMERA_CAPTURE_SIZE[0],
                            frameHeight=CAMERA_CAPTURE_SIZE[1],
                            frameRate=CAMERA_FPS)
        self.dedupe = dedupe
        self._prev_sig = None

    def read(self):
        ok = self.cam.read()
        img = self.cam.imageData
        if ok is False or img is None or getattr(img, "size", 0) == 0:
            return None
        if self.dedupe:
            # read() may hand back the previous frame if no new one has arrived;
            # skip exact repeats so one frame can't count twice toward confirmation.
            sig = img[::32, ::32].copy()
            if self._prev_sig is not None and sig.shape == self._prev_sig.shape \
                    and np.array_equal(sig, self._prev_sig):
                return None
            self._prev_sig = sig
        return img

    def terminate(self):
        self.cam.terminate()
