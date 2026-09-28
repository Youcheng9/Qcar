# Import necessary libraries
import signal
import sys
import os
import numpy as np
import time
import cv2
from pal.products.qcar import QCar
from lidarThread import Lidar  # Assuming Lidar class is imported correctly
from pal.utilities.math import *
from pal.utilities.vision import Camera2D
import tensorflow as tf


gpus = tf.config.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print(f"\nGPU: {gpu.name}, Details: {details}")
else:
    print("No GPU detected")

device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print("Model Utilizing: " + device + "\n")

# Environment Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
myCar = QCar(readMode=0)
lidar_device = Lidar(type='RPLidar')

#Test Changes
# Initialize cameras - assuming camera initialization remains as before
camera_front = Camera2D(cameraId="3", frameWidth=420, frameHeight=220, frameRate=30) 

# CAR PARAMS: set up +- increment and maximum
deceleration_increment = 0.00001    
throttle_increment = 0.005
steering_increment = 0.005
max_throttle = 0.1
min_throttle = -0.1
max_steering = 0.5
min_steering = -0.5
MINIMUM_THROTTLE_EPSILON = 0.0000001
counter = 0

blink_state = False
last_blink_time = 0
blink_interval = 0.2

# Initialize car control values
throttle = 0.0
steering = 0.0
LEDs = np.zeros(8)

# Steering classes for model prediction via "One Hot Encoding"
steering_classes = [-0.5, -0.4375, -0.375, -0.3125, -0.25, -0.1875, -0.125, -0.0625, 0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.5]

stop_threads = False
obstacle_detected = False

# Load the ML driving model
def loadModel(filename):
    '''driving_model = tf.lite.Interpreter(model_path=filename)
    driving_model.allocate_tensors()
    return driving_model'''
    try:
        # Load TensorFlow Lite GPU delegate
        gpu_delegate = tf.lite.experimental.load_delegate("libtensorflowlite_gpu_delegate.so")
        driving_model = tf.lite.Interpreter(model_path=filename, experimental_delegates=[gpu_delegate])
        print("Using GPU Delegate for TensorFlow Lite")
    except Exception as e:
        print("GPU delegate could not be loaded, falling back to CPU:", e)
        driving_model = tf.lite.Interpreter(model_path=filename)

    driving_model.allocate_tensors()
    return driving_model

# Using the camPreview function from AutonomousLaneLoop
def camPreview(camIDs, steering=0, throttle=0, catalog=None):
    if camIDs is not None:
        if "front" in camIDs:
            camera_front.read()
            if camera_front is not None:
                return camera_front.imageData
    return None

def signal_handler(sig , frame):
    global lidar_device , myCar
    lidar_device.terminate()
    myCar.terminate()
    print("Program terminated")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
# Get steering prediction from ML model
def getModelSteering(model, input_details, output_details):
    global counter
    image_cap_np = camPreview(camIDs=["front"])
    height, width = image_cap_np.shape[:2]
    image_cap_np = image_cap_np[int(height*.5):int(height*.85), :]
    image_cap_grayscale = cv2.cvtColor(image_cap_np, cv2.COLOR_BGR2GRAY)
    image_cap_grayscale = image_cap_grayscale[np.newaxis, 10:, :, np.newaxis]
    image_cap_grayscale = image_cap_grayscale.astype('float32')
    
    model.set_tensor(input_details[0]['index'], image_cap_grayscale)
    model.invoke()
    
    predictions = model.get_tensor(output_details[0]['index'])
    class_assign = np.argmax(predictions)
    
    counter += 1
    print(f"In Model Mode  Counter = {counter}")
    return steering_classes[class_assign]

def blinker(steering):
    global blink_state, last_blink_time, LEDs

    now = time.time()

    # turn all off first
    LEDs[0] = 0   # left
    LEDs[1] = 0   # right

    if abs(steering) > 0.1:
        if now - last_blink_time >= blink_interval:
            blink_state = not blink_state
            last_blink_time = now

        if steering > 0:
            LEDs[0] = 1 if blink_state else 0
        elif steering < 0:
            LEDs[1] = 1 if blink_state else 0
    else:
        blink_state = False

# Main control loop
def main():
    global stop_threads, throttle, steering, obstacle_detected
    
    # Load the ML model
    print("Loading ML model...")
    driving_model = loadModel("Models/Mar12Class.tflite")
    input_details = driving_model.get_input_details()
    output_details = driving_model.get_output_details()
    print("ML model loaded successfully")
    
    print("Main control loop started. Press Ctrl+C to exit.")
    running = True
    
    while running:
        # LiDAR functionality integrated directly
        lidar_device.read()
        
        if lidar_device.distances is None or lidar_device.angles is None or lidar_device.distances.size == 0:
            print("LiDAR data is empty! Skipping iteration.")
            continue
        
        front_view = (lidar_device.angles >= np.deg2rad(70)) & (lidar_device.angles <= np.deg2rad(110))
        front_distances = lidar_device.distances[np.nonzero(front_view)]
        front_distances = front_distances[front_distances > 0]
        
        danger_threshold = 0.6
        
        if len(front_distances) > 0 and np.any(front_distances < danger_threshold):
            if not obstacle_detected:
                print("Obstacle detected! Switching to LiDAR avoidance mode")
                obstacle_detected = True
                original_steering = steering
        
            left_view = (lidar_device.angles >= np.deg2rad(60)) & (lidar_device.angles <= np.deg2rad(90))
            right_view = (lidar_device.angles >= np.deg2rad(90)) & (lidar_device.angles <= np.deg2rad(120))
            
            left_distances = lidar_device.distances[left_view]
            right_distances = lidar_device.distances[right_view]
            
            left_distances = left_distances[~np.isnan(left_distances)]
            right_distances = right_distances[~np.isnan(right_distances)]
            
            left_clearance = np.nanmean(left_distances) if left_distances.size > 0 else np.inf
            right_clearance = np.nanmean(right_distances) if right_distances.size > 0 else np.inf
            
            print(f"Left Clearance: {left_clearance}, Right Clearance: {right_clearance}")
            
            if left_clearance > right_clearance + 0.1:
                print("Steering left to avoid obstacle")
                new_steering = max_steering
                throttle = 0.07
            elif right_clearance > left_clearance + 0.1:
                print("Steering right to avoid obstacle")
                new_steering = min_steering
                throttle = 0.07
            else:
                new_steering = 0.0
                throttle = 0.08
            
            steering = 0.7 * steering + 0.3 * new_steering
            
        else:
            if obstacle_detected:
                print("No obstacle detected! Switching back to ML model navigation")
                obstacle_detected = False
            
            if not stop_threads:
                new_steering = getModelSteering(driving_model, input_details, output_details)
                steering = 0.8 * steering + 0.3 * new_steering
                throttle = 0.075
        
        # Apply control commands to the car
        blinker(steering)
        myCar.write(throttle=throttle, steering=steering, LEDs=LEDs)
        time.sleep(0.1)

    
   

if __name__ == '__main__':
    main()
