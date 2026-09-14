#!/usr/bin/env python3

from flask import Flask, Response
from pal.utilities.vision import Camera3D
import cv2
import numpy as np
import time
import argparse

app = Flask(__name__)

camera = None

WIDTH = 640
HEIGHT = 480
FPS = 30
JPEG_QUALITY = 60


def read_realsense_rgb():
    """
    Tries common PAL Camera3D RGB read methods/buffers.
    Different PAL versions may use slightly different names.
    """
    global camera

    # Try common read methods
    if hasattr(camera, "read_RGB"):
        camera.read_RGB()
    elif hasattr(camera, "read"):
        camera.read()
    else:
        raise RuntimeError("Camera3D has no read_RGB() or read() method")

    # Try common RGB buffer names
    possible_attrs = [
        "imageBufferRGB",
        "imageData",
        "imageDataRGB",
        "rgbData",
    ]

    frame = None

    for attr in possible_attrs:
        if hasattr(camera, attr):
            value = getattr(camera, attr)
            if value is not None:
                frame = value
                break

    if frame is None:
        return None

    frame = np.asarray(frame, dtype=np.uint8)
    frame = np.ascontiguousarray(frame)

    # If image is RGB, convert to BGR for OpenCV JPEG encoding.
    # If it is already BGR, this may swap colors, but the stream will still work.
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame


def generate_frames():
    frame_delay = 1.0 / FPS

    while True:
        start_time = time.time()

        try:
            frame = read_realsense_rgb()

            if frame is None:
                continue

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            if not success:
                continue

            jpg_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpg_bytes +
                b"\r\n"
            )

            elapsed = time.time() - start_time
            sleep_time = frame_delay - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

        except Exception as e:
            print("Frame error:", e)
            time.sleep(0.1)


@app.route("/")
def index():
    return """
    <html>
        <head>
            <title>QCar RealSense Stream</title>
        </head>
        <body>
            <h1>QCar RealSense Stream</h1>
            <img src="/video" width="840">
        </body>
    </html>
    """


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def main():
    global camera, WIDTH, HEIGHT, FPS, JPEG_QUALITY

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=60)
    args = parser.parse_args()

    WIDTH = args.width
    HEIGHT = args.height
    FPS = args.fps
    JPEG_QUALITY = args.quality

    print("Opening Intel RealSense using PAL Camera3D...")

    camera = Camera3D(
        mode="RGB",
        frameWidthRGB=WIDTH,
        frameHeightRGB=HEIGHT,
        frameRateRGB=FPS
    )

    print(f"Starting HTTP RealSense stream on port {args.port}")
    print("Open this on laptop:")
    print(f"http://192.168.1.132:{args.port}/video")
    print()
    print("Preview page:")
    print(f"http://192.168.1.132:{args.port}/")

    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=False
    )


if __name__ == "__main__":
    main()
