import argparse
import threading
import sys
import socket
import os
import numpy as np
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import time
from pal.products.qcar import QCar
from pal.utilities.vision import Camera2D
from lidar2 import Lidar
import payload

# Environment Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Lidar catalog file
catalog = open("lidar-catalogs.txt", "w")
catalog.write("Count, Angles, Distances\n")
global_count = 0

# Car and Camera setup
myCar = QCar(readMode=0)
camera_front = Camera2D(cameraId="3", frameWidth=420, frameHeight=220, frameRate=30)
max_throttle = 0.075
min_throttle = -0.075
max_steering = 0.5
min_steering = -0.5
LEDs = np.array([0, 0, 0, 0, 0, 0, 0, 0])

# Control variables
steering = 0
throttle = 0
reverse = False
PORT = 38821
stop_threads = False

# Helper function for polar to cartesian conversion
def polar_to_cartesian(angles, distances):
    x = distances * np.cos(angles)  # angles are already radians
    y = distances * np.sin(angles)
    return x, y

# LiDAR function to collect and log data
def lidar_collection():
    global global_count, stop_threads
    lidar_device = Lidar(type='RPLidar')
    plt.ion()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    try:
        while not stop_threads:
            plt.cla()
            lidar_device.read()

            if lidar_device.distances is not None and lidar_device.angles is not None:
                ax.scatter(lidar_device.angles, lidar_device.distances, marker='.')
                ax.set_theta_zero_location("W")
                ax.set_theta_direction(-1)
                ax.set_title("LiDAR Scan Data", va='bottom')
                plt.pause(0.1)

                x, y = polar_to_cartesian(lidar_device.angles, lidar_device.distances)
                data = f"{global_count}, {x.tolist()}, {y.tolist()}\n"
                catalog.write(data)
                global_count += 1
    finally:
        plt.ioff()
        lidar_device.terminate()
        catalog.close()

# Drive function for controlling the car
def drive():
    global throttle, steering, reverse, stop_threads
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(('', PORT))

        while not stop_threads:
            data = s.recvfrom(100)[0].decode('utf-8')
            if not data:
                continue

            packet = payload.payload_handler(data)
            buffer = []
            try:
                # Process the packet data as in your original code
                # Update throttle, steering, reverse based on packet events
                
                if packet.read(buffer, 4) == -1 or \
                   packet.read(buffer, 4) == -1 or \
                   packet.read(buffer, 8) == -1 or \
                   packet.read(buffer, 8) == -1:
                    print("Warning: Packet Length Too short")
                    continue
                
                event = int(buffer[1])
                print("throttle: " + str(throttle))
                #print(event)

                if event == 1536:
                    #IF Axis is Steering Wheel
                    #print(float(buffer[2]))
                    print(steering)
                    if float(buffer[2]) == 0:
                        steer = (-1* float(buffer[3])) / 1.5
                        if abs(steering - steer) < 0.05:
                            continue

                        if (steering == min_steering and steer < min_steering
                            or
                            steering == max_steering and steer > max_steering):
                            continue

                        if steer < 0:
                            steering = max(steer, min_steering)
                        else:
                            steering = min(steer, max_steering)
                        
                    #IF Axis is Throttle
                    elif float(buffer[2]) == 1:
                        
                        #th = 0.6 * ((abs(float(buffer[3]) -1 ) /2 ) * 0.2)
                        th = (float(buffer[3])) / 2 - 0.5
                        throttle = th * max_throttle
                        #if th < 0:
                            #throttle = max(th, min_throttle)
                        #else:
                          #throttle = min(th, max_throttle)
                    
                    #IF Axis is Brake
                    elif float(buffer[2]) == 2:
                        continue 
                        #Implement Brake algorithm

                if event == 1539:
                    #print(float(buffer[2]))
                    if float(buffer[2]) == 5:
                        print("Reverse Triggered")
                        reverse = True
                    elif float(buffer[2]) == 4:
                        print("Forward Triggered")
                        reverse = False
                    elif float(buffer[2]) == 0:
                        steering = 0
                        throttle = 0
                        reverse = False
                    elif float(buffer[2]) == 11:
                        # Reverse Gear selected. 
                        print("Reverse Gear Selected")
                        min_throttle = -.075
                        max_throttle = .075
                        reverse = True
                    elif float(buffer[2]) == 12:
                        # First Gear Selected 
                        print("1st Gear Selected")
                        max_throttle = .075
                        reverse = False
                    elif float(buffer[2]) == 13:
                        # First Gear Selected 
                        print("2nd Gear Selected")
                        max_throttle = .125
                        reverse = False
                    elif float(buffer[2]) == 14:
                        # First Gear Selected 
                        print("3rd Gear Selected")
                        max_throttle = .2
                        reverse = False
                    elif float(buffer[2]) == 15:
                        # First Gear Selected 
                        print("4th Gear Selected")
                        max_throttle = .225
                        reverse = False
                    elif float(buffer[2]) == 16:
                        # First Gear Selected 
                        print("5th Gear Selected")
                        max_throttle = .3
                        reverse = False
                    elif float(buffer[2]) == 17:
                        # First Gear Selected 
                        print("6th Gear Selected")
                        max_throttle = .35
                        reverse = False
                    elif float(buffer[2]) == 6:
                        LEDs[6] = 1 if LEDs[6] == 0 else 0
                        LEDs[7] = 1 if LEDs[7] == 0 else 0
                        LEDs[4] = 1 if LEDs[4] == 0 else 0
                    
                        
            

                if reverse:
                    if throttle > 0:
                        throttle *= -1
                else:
                    throttle = abs(throttle)
                    
                    
                

                myCar.write(throttle=throttle, steering=steering, LEDs=LEDs)
            except Exception as e:
                print("Invalid Packet Size:", e)

    print("Terminated Driving")

# Camera preview function
def camPreview():
    while not stop_threads:
        camera_front.read()
        if camera_front is not None:
            cv2.imshow("Camera Front", camera_front.imageData)
        if cv2.waitKey(100) == 27:  # exit on ESC
            break
    camera_front.terminate()
    cv2.destroyWindow("Camera Front")

# Main function to start the threads
def main():
    global stop_threads

    try:
        # Threads for driving, camera, and LiDAR
        drive_thread = threading.Thread(target=drive)
        # cam_thread = threading.Thread(target=camPreview)
        lidar_thread = threading.Thread(target=lidar_collection)

        drive_thread.start()
        # cam_thread.start()
        lidar_thread.start()

        # Wait for threads to complete
        drive_thread.join()
        # cam_thread.join()
        lidar_thread.join()
    except KeyboardInterrupt:
        stop_threads = True
        myCar.terminate()
        sys.exit()

if __name__ == '__main__':
    main()
