"""Small shared helpers: angle wrapping and LED patterns."""

import numpy as np

from .config import LED_LEFT_IND, LED_RIGHT_IND, LED_BRAKE, LED_HEAD


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def leds(left=False, right=False, brake=False, head=True):
    arr = np.zeros(8)
    if left:
        arr[list(LED_LEFT_IND)] = 1
    if right:
        arr[list(LED_RIGHT_IND)] = 1
    if brake:
        arr[list(LED_BRAKE)] = 1
    if head:
        arr[list(LED_HEAD)] = 1
    return arr
