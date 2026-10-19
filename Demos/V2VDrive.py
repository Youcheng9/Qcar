"""Autonomous lane following plus V2V.

Runs the SelfDrivingTest.py steering model, shares this car's state with the
other car, and lowers throttle when the other car is braking or reports an
obstacle. If the link is lost the car keeps lane-following on its own
(--stop-on-link-loss changes that to a full stop).

    py3 V2VDrive.py --id qcar1 [--peer <other car IP>] [--stop-on-link-loss]

Brake on the other car's behalf for testing with v2v_demo.py --brake-after.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "V2V"))

import numpy as np

import SelfDrivingTest as sdt  # reuse the model, camera, and QCar helpers
from pal.products.qcar import QCar
from pal.utilities.vision import Camera2D
from v2v import V2VLink, limit_throttle

TURN_THRESHOLD = 0.1  # same threshold AutoLED.py uses for the blinkers


def turn_signal(steering):
    if steering > TURN_THRESHOLD:
        return "left"   # matches AutoLED.py: LEDs[0] is left for steering > 0
    if steering < -TURN_THRESHOLD:
        return "right"
    return "none"


def drive(args):
    leds = np.zeros(8, dtype=np.float64)
    qcar = camera = link = None
    previous_steering = 0.0
    period = 1.0 / sdt.LOOP_FREQUENCY
    last_reason = None

    try:
        model = sdt.SteeringModel(sdt.MODEL_PATH)
        qcar = QCar()
        camera = Camera2D(
            cameraId=sdt.CAMERA_ID, frameWidth=sdt.CAMERA_WIDTH,
            frameHeight=sdt.CAMERA_HEIGHT, frameRate=sdt.CAMERA_FPS)
        link = V2VLink(args.id, peer_ip=args.peer).start()
        print("V2V drive started as", args.id, "- Ctrl+C to stop.")

        while sdt.running:
            loop_start = time.time()

            frame = sdt.read_camera_frame(camera)
            predicted = model.predict(frame)
            steering = sdt.calculate_steering(predicted, previous_steering)
            previous_steering = steering

            peer = link.get_peer()
            throttle, reason = limit_throttle(sdt.AUTONOMOUS_THROTTLE, peer)
            if args.stop_on_link_loss and peer is None and link.ever_heard_peer():
                throttle, reason = 0.0, "link-lost"

            if reason != last_reason:  # log only on change
                print("V2V:", reason, "| peer:", peer)
                last_reason = reason

            link.update_state(
                throttle=throttle, steering=steering, mode="auto",
                brake=(throttle == 0.0), turn_signal=turn_signal(steering))
            sdt.write_qcar(qcar, throttle, steering, leds)

            remaining = period - (time.time() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        sdt.running = False
    except Exception as error:
        print("V2V drive error:", error)
    finally:
        print("Stopping QCar.")
        if link is not None:
            link.update_state(throttle=0.0, brake=True)
            time.sleep(0.2)  # let one last "braking" packet go out
            link.stop()
        if qcar is not None:
            sdt.stop_qcar(qcar, leds)
        for device in (camera, qcar):
            if device is not None:
                try:
                    device.terminate()
                except Exception:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="unique car name")
    parser.add_argument("--peer", help="other car's IP (default: broadcast)")
    parser.add_argument("--stop-on-link-loss", action="store_true")
    drive(parser.parse_args())
