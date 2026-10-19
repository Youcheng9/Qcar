# ===========================================================================================================
# File: FusionCollectionServer.py
# ===========================================================================================================
# Description:
#   DataCollectionServer.py + synchronized 2D LiDAR. Drive with the Logitech wheel (same TestDrive.py
#   client and port 38822) and record camera + LiDAR + steering/throttle/stop samples in the fusion.py
#   session format.
#
# Run on the QCar (from this folder):
#   sudo python3 FusionCollectionServer.py --tag normal
#
# Differences from DataCollectionServer.py:
#   - binds every interface, so there is no IP to edit (the PC client still needs the car's IP)
#   - samples at a fixed rate (--hz) instead of every 1000th loop
#   - brake pedal = stop label: holding it zeroes throttle and samples are recorded while stopped
#   - each session goes to data/<date_time>_<tag>/ ; no index_tracker.txt / images_catalogs.txt
#   - XBOX button or Ctrl+C stops and closes everything cleanly
# ===========================================================================================================
import argparse
import os
import socket
import sys
import threading
import time
from datetime import datetime

import cv2  # imported up front: a lazy first import would stall the drive loop mid-run
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path += [os.path.join(HERE, '..', 'LogitechWheelPrograms'), os.path.join(HERE, '..', 'LidarStuff')]

from pal.products.qcar import QCar
from pal.utilities.vision import Camera2D
from lidar2 import Lidar
import payload
import fusion

PORT = 38822
MAX_STEERING = 0.5
BRAKE_STOP = 0.2        # brake pedal travel (0..1) that counts as "stop"
# shifter button -> (max_throttle, reverse), same values as DataCollectionServer.py
GEARS = {11: (.075, True), 12: (.075, False), 13: (.125, False), 14: (.2, False),
         15: (.225, False), 16: (.3, False), 17: (.35, False)}

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='normal',
                    help='scenario: normal, obstacle, person, lighting, camera_fail, lidar_fail, ...')
parser.add_argument('--hz', type=float, default=10.0, help='samples per second')
parser.add_argument('--out', default=os.path.join(HERE, 'data'))
args = parser.parse_args()

camera = Camera2D(cameraId="3", frameWidth=420, frameHeight=220, frameRate=30)
lidar = Lidar(type='RPLidar')
car = QCar(readMode=0)
LEDs = np.array([0, 0, 0, 0, 0, 0, 0, 0])

running = True
scan = (None, None, None)        # latest (t, angles, distances) from the LiDAR thread
scan_lock = threading.Lock()


def lidar_loop():
    """Keep the newest scan. Only re-stamp when the data changes, so a stale/dead LiDAR shows up as
    a growing image_t - lidar_t instead of fresh-looking copies."""
    global scan
    last = None
    while running:
        lidar.read()
        d = np.array(lidar.distances, dtype=np.float32).ravel()
        if d.size and (last is None or not np.array_equal(d, last)):
            last = d
            with scan_lock:
                scan = (time.time(), np.array(lidar.angles, dtype=np.float32).ravel(), d)
        time.sleep(0.005)


def main():
    global running
    folder = os.path.join(args.out, datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + args.tag)
    writer = fusion.SessionWriter(folder, args.tag)
    print('Recording to', folder)

    steering = pedal = brake = 0.0
    max_throttle, reverse = .075, False
    period, next_sample = 1.0 / args.hz, time.time()
    threading.Thread(target=lidar_loop, daemon=True).start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', PORT))
    sock.settimeout(0.005)
    try:
        while running:
            try:
                packet = payload.payload_handler(sock.recvfrom(100)[0].decode('utf-8'))
                buffer = []
                # |Controller ID 4|Event 4|Event Dimension 8|Event Value 8|
                if any(packet.read(buffer, size) == -1 for size in (4, 4, 8, 8)):
                    print('Warning: Packet Length Too short')
                    continue
                event, dim, value = int(buffer[1]), int(float(buffer[2])), float(buffer[3])

                if event == 1536:                       # axis motion
                    if dim == 0:                        # wheel
                        steer = float(np.clip(-value / 0.75, -MAX_STEERING, MAX_STEERING))
                        if abs(steer - steering) >= 0.05 or abs(steer) == MAX_STEERING:
                            steering = steer
                    elif dim == 1:                      # throttle pedal: 1 released, -1 floored
                        pedal = (1 - value) / 2
                    elif dim == 2:                      # brake pedal, same range
                        brake = (1 - value) / 2
                elif event == 1539:                     # button down
                    if dim == 5:
                        print('Reverse Triggered')
                        reverse = True
                    elif dim == 4:
                        print('Forward Triggered')
                        reverse = False
                    elif dim == 0:                      # neutral
                        steering = pedal = 0.0
                        reverse = False
                    elif dim in GEARS:
                        max_throttle, reverse = GEARS[dim]
                        print('Gear', dim - 11 if dim > 11 else 'R', 'max_throttle', max_throttle)
                    elif dim == 6:
                        LEDs[[4, 6, 7]] = 1 - LEDs[[4, 6, 7]]
                    elif dim == 10:                     # XBOX button
                        print('SHUTTING DOWN !!')
                        break
            except socket.timeout:
                pass
            except (ValueError, UnicodeDecodeError) as e:
                print('Invalid Packet:', e)

            stop = brake > BRAKE_STOP
            throttle = 0.0 if stop else pedal * max_throttle * 0.4 * (-1 if reverse else 1)
            car.write(throttle=throttle, steering=steering, LEDs=LEDs)

            now = time.time()
            if now >= next_sample:
                next_sample = now + period
                if throttle != 0 or stop:
                    camera.read()
                    image_t = time.time()
                    image = camera.imageData
                    if image is not None:
                        image = image.copy()
                    with scan_lock:
                        lidar_t, angles, distances = scan
                    writer.add(now, image, image_t, angles, distances, lidar_t, steering, throttle, stop)
                    skew = image_t - lidar_t if lidar_t else float('nan')
                    print('%6d steer %+.3f thr %+.3f stop %d skew %.3fs%s' % (
                        writer.index - 1, steering, throttle, stop, skew,
                        '  <-- check sensors' if image is None or not skew < 0.25 else ''))
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        car.write(throttle=0, steering=0, LEDs=np.zeros(8))
        writer.close()
        sock.close()
        time.sleep(0.1)
        lidar.terminate()
        camera.terminate()
        car.terminate()
        print('Saved %d samples to %s' % (writer.index, folder))


if __name__ == '__main__':
    main()
