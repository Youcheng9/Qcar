"""Gyroscope diagnostic (the car does not drive). Answers three questions:

  1. How noisy is QCar.gyroscope at rest, on each axis?
  2. Units: is it rad/s (as the parking code assumes) or deg/s?
  3. Sign: is the z axis positive when the car turns LEFT (GYRO_Z_SIGN)?

Usage (on the car, same way you run the parking script):
    sudo python3 gyro_check.py

Phase 1 (5 s): leave the car completely still.
Phase 2 (15 s): turn the car by hand exactly 90 deg to the LEFT (counter-clockwise
seen from above), about 3-5 s for the turn, then hold it still. A square
tile/table edge helps you get 90 deg.
"""

import math
import time

import numpy as np

from pal.products.qcar import QCar

from parking.config import GYRO_CAL_MAX_STD, GYRO_Z_SIGN, SAMPLE_RATE
from parking.utils import leds

STILL_S = 5.0
TURN_S = 15.0


def read_gyro(car):
    car.read_write_std(throttle=0.0, steering=0.0, LEDs=leds(brake=True))
    return np.array(car.gyroscope, dtype=float).reshape(-1)[:3]


def main():
    car = QCar(readMode=0, frequency=SAMPLE_RATE)
    try:
        print("Phase 1: keep the car completely still for {:.0f} s...".format(STILL_S))
        samples, times = [], []
        t_end = time.monotonic() + STILL_S
        while time.monotonic() < t_end:
            samples.append(read_gyro(car))
            times.append(time.monotonic())
            time.sleep(1.0 / SAMPLE_RATE)
        s = np.array(samples)
        dts = np.diff(times)
        print("  {} samples, {:.1f} ms average between reads".format(len(s), 1000 * dts.mean()))
        print("  first 10 z values: {}".format(np.round(s[:10, 2], 4)))
        for i, axis in enumerate("xyz"):
            print("  {}: mean {:+.4f}  std {:.4f}  min {:+.4f}  max {:+.4f}".format(
                axis, s[:, i].mean(), s[:, i].std(), s[:, i].min(), s[:, i].max()))
        same = np.mean(np.all(np.diff(s, axis=0) == 0, axis=1))
        print("  {:.0f}% of reads were identical to the previous one".format(100 * same))
        # skip the first 10 samples in case the sensor starts at zero
        bias = s[10:, 2].mean()
        z_std = s[10:, 2].std()

        print("\nPhase 2: within {:.0f} s, turn the car 90 deg LEFT by hand, then hold it still..."
              .format(TURN_S))
        angle = 0.0
        prev = time.monotonic()
        t_end = prev + TURN_S
        next_print = prev + 1.0
        while time.monotonic() < t_end:
            gz = read_gyro(car)[2] - bias
            now = time.monotonic()
            angle += gz * (now - prev)
            prev = now
            if now >= next_print:
                print("  integrated z: {:+8.3f}".format(angle))
                next_print += 1.0
            time.sleep(1.0 / SAMPLE_RATE)
    finally:
        car.terminate()

    print("\n=== Results ===")
    print("Integrated z after a 90 deg left turn: {:+.3f}".format(angle))
    a = abs(angle)
    if 1.2 < a < 2.0:
        units = "rad/s"
        print("-> about 1.57, so the gyro is in rad/s (what the code assumes)")
    elif 70 < a < 110:
        units = "deg/s"
        print("-> about 90, so the gyro is in DEG/S. The parking code needs a conversion.")
    else:
        units = None
        print("-> neither ~1.57 (rad/s) nor ~90 (deg/s). Did the turn happen within the "
              "time, and was it about 90 deg? Run it again.")
    if units:
        sign = 1 if angle > 0 else -1
        print("-> a left turn gave a {} value, so GYRO_Z_SIGN should be {:+d} (config has {:+d}){}"
              .format("positive" if sign > 0 else "negative", sign, GYRO_Z_SIGN,
                      "" if sign == GYRO_Z_SIGN else "  <-- change it"))
        z_std_rad = z_std if units == "rad/s" else math.radians(z_std)
        print("-> noise at rest: {:.4f} rad/s (limit GYRO_CAL_MAX_STD = {:.4f}){}".format(
            z_std_rad, GYRO_CAL_MAX_STD,
            "" if z_std_rad <= GYRO_CAL_MAX_STD else "  <-- still too noisy"))
    print("\nCopy this whole output back to Claude.")


if __name__ == "__main__":
    main()
