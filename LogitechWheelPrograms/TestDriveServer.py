import argparse

import time
import subprocess
import threading
import sys
import socket
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import numpy as np
import cv2
from pal.products.qcar import QCar, QCarRealSense
from pal.utilities.vision import Camera2D
from datetime import datetime



import payload

stopthread = False

myCar = QCar(readMode=0)
max_throttle = 0.075
min_throttle = -0.075
max_steering = 0.5
min_steering = -0.5
LEDs = np.array([0, 0, 0, 0, 0, 0, 0, 0])

steering = 0
throttle = 0
currentGear = 1
speeds = [0, .075, .125, .2, .225, .3, .35]
reverse = False

PORT = 38822  # Port to listen on (non-privileged ports are > 1023)
camera_right = Camera2D(cameraId="0",frameWidth=420,frameHeight=220,frameRate=30)
camera_back = Camera2D(cameraId="1",frameWidth=420,frameHeight=220,frameRate=30)
camera_left = Camera2D(cameraId="2",frameWidth=420,frameHeight=220,frameRate=30)
camera_front = Camera2D(cameraId="3",frameWidth=420,frameHeight=220,frameRate=30)

def get_wifi_networks():
   while(True):
    """Gets the list of available Wi-Fi networks."""

    command = "nmcli device wifi list"
    output = subprocess.check_output(command, shell=True)
    decoded_data = output.decode('utf-8')
    lines = decoded_data.splitlines()
    for line in lines:
            print(line.strip())
    print("\n" + "-" * 50)
    print("\n")
    time.sleep(2)

def drive():   
    print("Driving Starting...")
    left_indicator = 0 
    right_indicator = 0
    rear_cam = False
    right_cam = False
    left_cam = False
    global throttle, steering, reverse, currentGear, speeds, max_throttle, min_throttle
    
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(('0.0.0.0', PORT))
        global stopthread
        
        while True:
            if stopthread:
                break

            data = s.recvfrom(100)[0].decode('utf-8')
            if not data:
                pass

            packet = payload.payload_handler(data)
            buffer = []
            try:
                '''
                /*##################################################################################*/
                /*|    Controller ID    |    Event    |    Event Dimension     |    Event Value    |*/
                /*|        4 Bit        |    4 Bit    |         8 Bit          |       4 Bit       |*/
                /*##################################################################################*/
                '''
                
                '''
                Read Payload Data
                '''
                if packet.read(buffer, 4) == -1 or \
                   packet.read(buffer, 4) == -1 or \
                   packet.read(buffer, 8) == -1 or \
                   packet.read(buffer, 8) == -1:
                    print("Warning: Packet Length Too short")
                    continue
                
                event = int(buffer[1])
                #print("throttle: " + str(throttle))
                #print(event)

                if event == 1536:
                    #IF Axis is Steering Wheel
                    #print(float(buffer[2]))
                    #print(steering)
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
                        toggle_led(4)
                        continue 
                        #Implement Brake algorithm

                if event == 1539:
                    #print(float(buffer[2]))
                    if float(buffer[2]) == 5:
                        reverse = False
                        currentGear = currentGear - 1 if currentGear > 1 else currentGear
                        print("Gear: " + str(currentGear))
                        max_throttle = speeds[currentGear]
                    elif float(buffer[2]) == 4:
                        currentGear = currentGear + 1 if currentGear < 6 else currentGear
                        print("Gear: " + str(currentGear))
                        max_throttle = speeds[currentGear]
                        reverse = False
                    elif float(buffer[2]) == 0:
                        print("Rear Cam Flipped")
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
                        LEDs[0] = 1
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
                    elif float(buffer[2]) == 1:
                        print("Right Cam Flipped")
                    elif float(buffer[2]) == 2:
                        print("REVERSE ACTIVATED")
                        min_throttle = -.075
                        max_throttle = .075
                        reverse = True
                if reverse:
                    if throttle > 0:
                        throttle *= -1
                else:
                    throttle = abs(throttle)
                myCar.write(throttle=throttle, steering=steering, LEDs = LEDs)
                
                # Outputs steering values
                # print(steering)
                    

            except Exception as e:
                print("Invalid Packet Size")
                print(e)

    print("Terminated Driving")

def toggle_led(ledNum):
    try:
        while True:
            for i in range(8):
                if i == ledNum:
                    LEDs[i] = 1
                else:
                    LEDs[i] = 0
                myCar.write(throttle=0, steering=0, LEDs=LEDs)
                time.sleep(0.5)

    finally:
        myCar.write(throttle=0, steering=0, LEDs=np.zeros(8))
    
def identify_lane(frame):

    # remove the QCAR bumper from view
    

    #gray_scaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    return frame
    
def camPreview(camIDs = ["front"]):

    while True:
        if "front" in camIDs:
            camera_front.read()
            if camera_front is not None:
                #cv2.namedWindow("Camera Front", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Camera Front", 960, 540)
                image = camera_front.imageData
                image_scaled = cv2.resize(image, (960, 540))
                #img = detect_lanes(image)
                cv2.imshow("Camera Front", image_scaled)
                
                frame = image
                height, width = frame.shape[:2]
                cropped_height = int(height*0.85)
                frame = frame[:cropped_height,:]

                # store image size
                height, width = frame.shape[:2]

                # crop the field of view
                #poly_shape = np.array([(0,height),(int(width*0.5),0),(int(width*0.5),0),(width,height),],dtype=np.int32)
                #cv2.fillPoly(mask,[poly_shape],(255))
                #cv2.fillPoly(mask,ellipse,(255))

                #return frame 
                # take img hsv color space
                hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # lane color range (orange)
                opaque_orange = np.array([255,204,204])
                saturated_orange = np.array([153,76,0])
                
                # saturate the lane color
                orange_mask = cv2.inRange(hsv_image,opaque_orange,saturated_orange)
                orange_region = cv2.bitwise_and(frame,frame,mask=orange_mask)
                hsv_image[:,:,1] = np.clip(hsv_image[:,:,1]*1.25,0,255)
                hsv_image[:,:,2] = np.clip(hsv_image[:,:,2]*1.25,0,255)

                # convert back 
                saturated_frame = cv2.cvtColor(hsv_image,cv2.COLOR_HSV2BGR)
                
                # blurr the background + apply edge detection
                blurred_image = cv2.GaussianBlur(saturated_frame,(3,3),1.5)
                edges = cv2.Canny(blurred_image,50,150)
                mask = np.zeros((height,width),dtype=np.uint8)
                ellipse = cv2.ellipse(mask,(width//2,height -50),(int(width//2),int(height//4)),0,0,360,255, -20)
                frame = cv2.bitwise_and(edges,edges,mask=mask)
                
                cv2.imshow("reynaTransform", frame)
                
        if "back" in camIDs:
            camera_back.read()
            if camera_back is not None:
                
                cv2.imshow("Camera Back", camera_back.imageData)
        key = cv2.waitKey(100)
        if key == 27:  # exit on ESC
            camera_front.terminate()
            cv2.destroyWindow("Camera Front")



keyControl = threading.Thread(target=drive)
cameraAccess = threading.Thread(target=camPreview,args=[["front"]])


def main():
    global keyControl, cameraAccess
    keyControl = threading.Thread(target=drive)
    #cameraAccess = threading.Thread(target=camPreview,args=[["front"]])
    #rear_camera_view = threading.Thread(target=camPreview,args=["back"])
    network = threading.Thread(target=get_wifi_networks)
    
    try:
        keyControl.start()
        #cameraAccess.start()
        network.start()
    except KeyboardInterrupt:
            myCar.terminate()
            # exitPygame
            pygame.quit()
            sys.exit()


    

if __name__ == '__main__':
    main()
