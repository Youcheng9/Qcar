import signal
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

# Use the same Quanser imports as the previous team's working code.
from pal.products.qcar import QCar
from pal.utilities.vision import Camera2D


# ============================================================
# Configuration
# ============================================================

#MODEL_PATH = "Models/qcar_steering_tf231.tflite"
MODEL_PATH = "Models/rightTurn.tflite"

CAMERA_ID = "3"
CAMERA_WIDTH = 420
CAMERA_HEIGHT = 220
CAMERA_FPS = 30

MODEL_WIDTH = 420
MODEL_HEIGHT = 67

# Must match train_model.py.
CROP_START_PERCENT = 0.55

MIN_STEERING = -0.5
MAX_STEERING = 0.5

# Begin with 0.0 while checking predictions.
AUTONOMOUS_THROTTLE = 0.07

# After confirming predictions, try:
# AUTONOMOUS_THROTTLE = 0.015

STEERING_SMOOTHING = 0.25
MAX_STEERING_CHANGE = 0.04

LOOP_FREQUENCY = 30.0
PRINT_PREDICTIONS = True


# ============================================================
# Global shutdown state
# ============================================================

running = True


def request_shutdown(signum=None, frame=None):
    global running
    running = False


signal.signal(signal.SIGINT, request_shutdown)
signal.signal(signal.SIGTERM, request_shutdown)


# ============================================================
# TensorFlow Lite model
# ============================================================

class SteeringModel:
    def __init__(self, model_path):
        print("TensorFlow version:", tf.__version__)
        print("Loading model:", model_path)

        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=2
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()[0]
        )

        self.output_details = (
            self.interpreter.get_output_details()[0]
        )

        self.input_shape = tuple(
            self.input_details["shape"]
        )

        self.input_dtype = self.input_details["dtype"]
        self.output_dtype = self.output_details["dtype"]

        print("Model input shape:", self.input_shape)
        print("Model input dtype:", self.input_dtype)
        print(
            "Model output shape:",
            self.output_details["shape"]
        )
        print("Model output dtype:", self.output_dtype)

        expected_shape = (
            1,
            MODEL_HEIGHT,
            MODEL_WIDTH,
            1
        )

        if self.input_shape != expected_shape:
            raise ValueError(
                "Unexpected model input shape: {}. "
                "Expected: {}".format(
                    self.input_shape,
                    expected_shape
                )
            )

    def preprocess(self, frame):
        if frame is None:
            raise RuntimeError("Camera returned no frame.")

        if frame.size == 0:
            raise RuntimeError(
                "Camera returned an empty frame."
            )

        frame_height = frame.shape[0]

        crop_start = int(
            frame_height * CROP_START_PERCENT
        )

        cropped = frame[
            crop_start:,
            :
        ]

        if cropped.ndim == 3:
            grayscale = cv2.cvtColor(
                cropped,
                cv2.COLOR_BGR2GRAY
            )
        else:
            grayscale = cropped

        resized = cv2.resize(
            grayscale,
            (MODEL_WIDTH, MODEL_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        normalized = (
            resized.astype(np.float32) / 255.0
        )

        model_input = np.expand_dims(
            normalized,
            axis=-1
        )

        model_input = np.expand_dims(
            model_input,
            axis=0
        )

        return self.convert_input_dtype(model_input)

    def convert_input_dtype(self, model_input):
        if self.input_dtype == np.float32:
            return model_input.astype(np.float32)

        scale, zero_point = (
            self.input_details["quantization"]
        )

        if scale == 0:
            raise ValueError(
                "Invalid input quantization scale."
            )

        quantized = (
            model_input / scale + zero_point
        )

        if self.input_dtype == np.uint8:
            quantized = np.clip(
                quantized,
                0,
                255
            )

        elif self.input_dtype == np.int8:
            quantized = np.clip(
                quantized,
                -128,
                127
            )

        return quantized.astype(
            self.input_dtype
        )

    def convert_output(self, output):
        value = float(output.reshape(-1)[0])

        if self.output_dtype in (
            np.uint8,
            np.int8
        ):
            scale, zero_point = (
                self.output_details["quantization"]
            )

            if scale != 0:
                value = (
                    value - zero_point
                ) * scale

        return value

    def predict(self, frame):
        model_input = self.preprocess(frame)

        self.interpreter.set_tensor(
            self.input_details["index"],
            model_input
        )

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details["index"]
        )

        return self.convert_output(output)


# ============================================================
# Camera helper
# ============================================================

def read_camera_frame(camera):
    result = camera.read()

    # Some Quanser versions return the image directly.
    if isinstance(result, np.ndarray):
        return result.copy()

    # Other versions return True/False and update imageData.
    possible_buffers = [
        "imageData",
        "imageDataRGB",
        "imageBufferRGB",
        "rgbData"
    ]

    for buffer_name in possible_buffers:
        if hasattr(camera, buffer_name):
            frame = getattr(
                camera,
                buffer_name
            )

            if isinstance(frame, np.ndarray):
                return frame.copy()

    raise RuntimeError(
        "Could not obtain Camera2D image buffer. "
        "Check how Camera2D is read in the old program."
    )


# ============================================================
# Steering filtering
# ============================================================

def calculate_steering(
    prediction,
    previous_steering
):
    prediction = float(
        np.clip(
            prediction,
            MIN_STEERING,
            MAX_STEERING
        )
    )

    smoothed = (
        STEERING_SMOOTHING * prediction
        + (1.0 - STEERING_SMOOTHING)
        * previous_steering
    )

    steering_change = (
        smoothed - previous_steering
    )

    steering_change = float(
        np.clip(
            steering_change,
            -MAX_STEERING_CHANGE,
            MAX_STEERING_CHANGE
        )
    )

    steering = (
        previous_steering
        + steering_change
    )

    return float(
        np.clip(
            steering,
            MIN_STEERING,
            MAX_STEERING
        )
    )


# ============================================================
# QCar helpers
# ============================================================

def write_qcar(
    qcar,
    throttle,
    steering,
    leds
):
    try:
        qcar.write(
            throttle,
            steering,
            leds
        )

    except TypeError:
        qcar.write(
            throttle=throttle,
            steering=steering,
            LEDs=leds
        )


def stop_qcar(qcar, leds):
    try:
        write_qcar(
            qcar,
            0.0,
            0.0,
            leds
        )
    except Exception as error:
        print(
            "Could not send stop command:",
            error
        )


# ============================================================
# Main driving loop
# ============================================================

def drive():
    global running

    qcar = None
    camera = None

    leds = np.zeros(
        8,
        dtype=np.float64
    )

    previous_steering = 0.0

    loop_period = (
        1.0 / LOOP_FREQUENCY
    )

    try:
        model = SteeringModel(
            MODEL_PATH
        )

        qcar = QCar()

        camera = Camera2D(
            cameraId=CAMERA_ID,
            frameWidth=CAMERA_WIDTH,
            frameHeight=CAMERA_HEIGHT,
            frameRate=CAMERA_FPS
        )

        print()
        print("Autonomous driving started.")
        print(
            "Throttle:",
            AUTONOMOUS_THROTTLE
        )
        print("Press Ctrl+C to stop.")

        while running:
            loop_start = time.time()

            frame = read_camera_frame(
                camera
            )

            predicted_steering = (
                model.predict(frame)
            )

            steering = calculate_steering(
                predicted_steering,
                previous_steering
            )

            previous_steering = steering

            write_qcar(
                qcar,
                AUTONOMOUS_THROTTLE,
                steering,
                leds
            )

            if PRINT_PREDICTIONS:
                print(
                    "prediction={:+.5f} "
                    "steering={:+.5f}".format(
                        predicted_steering,
                        steering
                    )
                )

            elapsed = (
                time.time() - loop_start
            )

            remaining = (
                loop_period - elapsed
            )

            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        running = False

    except Exception as error:
        print()
        print("Autonomous driving error:")
        print(error)

    finally:
        print()
        print("Stopping QCar.")

        if qcar is not None:
            stop_qcar(
                qcar,
                leds
            )

        if camera is not None:
            try:
                camera.terminate()
            except Exception:
                pass

        if qcar is not None:
            try:
                qcar.terminate()
            except Exception:
                pass

        print("QCar stopped.")


if __name__ == "__main__":
    drive()
