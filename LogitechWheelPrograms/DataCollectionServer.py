# ===========================================================================================================
# File: DataCollectionServer.py
# ===========================================================================================================
# Project: Quanser Car "Class Hopper"
# Contributers: Alex Sanna, Matthew Baldivino, Reyna Nava
# Last Update: 07/08/2025 (by Reyna)
#  
# Description:
#   Run a QCar drive session to recieve images from QCar camera's and the respective steering and throttle
#   values.  
# Notes:
#   utilizes a UDP (utf-8) socket to recieve data packets from the QCAR
#   image_skipper: Controls the frequency of image collection
#   image_catalogs.txt: logs the image file names
#   index_tracker: logs the last image # collected
# ===========================================================================================================

# QCAR
from pal.utilities.vision import Camera2D
from pal.products.qcar import QCarRealSense
from pal.products.qcar import QCar, QCarRealSense
from pal.utilities.vision import Camera3D

# OS tasks + threading + communication
import os
import threading
import sys
import argparse
import socket
import payload

# image pre-processing
from datetime import datetime
import numpy as np
import cv2

# globals
PORT = 38822  # Port to listen on (non-privileged ports are > 1023)

# initial QCAR variables
global max_throttle, min_throttle, max_sterring, min_steering
max_throttle = 0.2
min_throttle = -0.2
max_steering = 0.5
min_steering = -0.5
LEDs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
steering = 0
throttle = 0
image_skipper = 0
reverse = False
stopthread = False


# setup + validate camera setup
CAMERA_FRONT = Camera2D(cameraId="3",frameWidth=420,frameHeight=220,frameRate=30)
#CAMERA_FRONT = Camera3D(mode='RGB', frameWidthRGB=1280,frameHeightRGB=720,frameRateRGB=30.0)
if CAMERA_FRONT is None:
    print("Error setting up camera front")
    exit()

# supress tensorflow log noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}

# create an instance of qcar
myCar = QCar(readMode=0)

# folder setup
folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
path = "./images/"
full_folder_path = os.path.join(path, folder_name)
os.makedirs(full_folder_path)  
print("Image directory created")


# = drive ===================================================================================================
# Description:
#   Start a socket connection with QCAR, verify data packets and write to the image_catalogs file to write
#   throttle and steering values
#
# Requires:
#   camPreview()
#
# Notes:
#   Handles Events from the TrueForce Steering Wheel, Throttle, Brake and shift knob gears 1-6
#   Error handling of packet size
#   Implements image skipping
#   Keeps track of image #
#
# Improvement Notes:
#   Brake Algorithm not implemented
# ===========================================================================================================
def drive():

    print("Driving Starting...")
    global throttle, steering, reverse, image_skipper, max_throttle, min_throttle

     #file management for memory purposes
    catalog = open(r"images_catalogs.txt", "a")
    tracker = open(r"index_tracker.txt", "r")
    global_index = tracker.read()
    global_index = int(global_index)
    global_count = global_index
    print("starting index: " + str(global_index))
    tracker.close()
    
    # setup socket connection
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        #s.bind(('10.110.129.185', PORT))
        s.bind(('192.168.1.132',PORT))
        s.setblocking(False)
        global stopthread

        while True:
            if stopthread:
                break

            try:
                #print("Waiting for UDP pckt")
                data = s.recvfrom(100)[0].decode('utf-8')
                #print("Recieived UDP pckt")
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
                    
                    # Read Payload Data
                    if packet.read(buffer, 4) == -1 or \
                    packet.read(buffer, 4) == -1 or \
                    packet.read(buffer, 8) == -1 or \
                    packet.read(buffer, 8) == -1:
                        print("Warning: Packet Length Too short")
                        continue
                    
                    event = int(buffer[1])

                    if event == 1536:
                        #IF Axis is Steering Wheel
                        #print("Button Pressed ID:", float(buffer[2]))
                        if float(buffer[2]) == 0:
                            steer = (-1* float(buffer[3])) / 0.75
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
                            
                            th = (float(buffer[3])) / 2 - 0.5
                            throttle = th * max_throttle * 0.4
                        
                        #IF Axis is Brake
                        elif float(buffer[2]) == 2:
                            continue 
                            #Implement Brake algorithm

                    if event == 1539:
                        if float(buffer[2]) == 5:           # Reverse Trigger
                            print("Reverse Triggered")
                            reverse = True
                        elif float(buffer[2]) == 4:
                            print("Forward Triggered")      # Forward Trigger
                            reverse = False
                        elif float(buffer[2]) == 0:         # Neutral
                            steering = 0
                            throttle = 0  
                            reverse = False
                        elif float(buffer[2]) == 11:        # Reverse Gear
                            print("Reverse Gear Selected")
                            min_throttle = -.075
                            max_throttle = .075
                            reverse = True
                        elif float(buffer[2]) == 12:        # First Gear
                            print("1st Gear Selected")
                            max_throttle = .075
                            reverse = False
                        elif float(buffer[2]) == 13:        # Second Gear
                            print("2nd Gear Selected")
                            max_throttle = .125
                            reverse = False
                        elif float(buffer[2]) == 14:        # Third Gear
                            print("3rd Gear Selected")
                            max_throttle = .2
                            reverse = False
                        elif float(buffer[2]) == 15:        # Fourth Gear
                            print("4th Gear Selected")
                            max_throttle = .225
                            reverse = False
                        elif float(buffer[2]) == 16:        # Fifth Gear
                            print("5th Gear Selected")
                            max_throttle = .3
                            reverse = False
                        elif float(buffer[2]) == 17:        # Sixth Gear
                            print("6th Gear Selected")
                            max_throttle = .35
                            reverse = False
                        elif float(buffer[2]) == 6:
                            LEDs[6] = 1 if LEDs[6] == 0 else 0
                            LEDs[7] = 1 if LEDs[7] == 0 else 0
                            LEDs[4] = 1 if LEDs[4] == 0 else 0
                            
                        elif float(buffer[2]) == 10:        # XBOX button
                            print("Recording last index!")
                            print("SHUTTING DOWN !!")
                            catalog.close()
                            tracker.close()
                            print("index upon shutdown: " + str(global_count))
                            tracker_update = open(r"index_tracker.txt", "w")
                            tracker_update.write(str(global_count))
                            tracker_update.close()
                            exit(0)

                    if reverse:
                        if throttle > 0:
                            throttle *= -1
                    else:
                        throttle = abs(throttle)
                        
                    myCar.write(throttle=throttle, steering=steering, LEDs = LEDs)
                    if throttle != 0 and image_skipper % 1000 == 0:
                        camPreview(["front"], global_count, steering, throttle, catalog)
                        global_count = global_count+1
                        
                    image_skipper += 1
                
                except Exception as e:
                    print("Invalid Packet Size")
                    print(e)
            except Exception as E:
                if throttle != 0  and image_skipper % 1000 == 0:
                    camPreview(["front"], global_count, steering, throttle, catalog)
                    global_count = global_count+1
                    
                image_skipper +=1

    print("Terminated Driving")
    
    
# = camPreview  =============================================================================================
# ===========================================================================================================
def camPreview(camIDs, global_count, steering, throttle, catalog):
    if int(global_count) >= 0:
        data = str(global_count) + ", "

        if "front" in camIDs:
            CAMERA_FRONT.read();
            if CAMERA_FRONT.imageData is not None:
                #snapshot function will take a picture
                snapshot(CAMERA_FRONT.imageData, global_count)
                img_name = 'sample_{}.jpg'.format(str(global_count))
                #data = str(global_count) + ", " + img_name + ", " + str(steering) + ", " + str(throttle) + "\n"
                data += (img_name + ", ")
                #print(data)
                #catalog.write(data)
                #global_count+=1

        if "back" in camIDs:
            camera_back.read()
            if camera_back is not None:
                print("back")
                #detect_lanes(camera_back.imageData)
                #cv2.imshow("Camera Back", camera_back.imageData)
        if "left" in camIDs:
            camera_left.read()
            if camera_left is not None:
                print("left")
                #detect_lanes(camera_left.imageData)
                #cv2.imshow("Camera Left", camera_left.imageData) 
        if "right" in camIDs:
            camera_right.read()
            if camera_right is not None:
                print("right")
                #detect_lanes(camera_right.imageData)
                #cv2.imshow("Camera Right", camera_right.imageData)   
        if 'depth' in camIDs:
            if depth_cam is not None: 
                max_distance = 10
                depth_cam.read_depth()
                #take depth image and store it as an array
                image = depth_cam.imageBufferDepthPX/max_distance
                arr_image = np.asanyarray(image)
                img_name = 'sample_' + str(global_count) + '.npy'
                data += (img_name + ", ")
                #save depth image to the specified path below
                path = "/media/378B-14FD/Depth_Images/"
                np.save(os.path.join(path, img_name), arr_image)
       
        #write to catalog
        if 'depth' not in camIDs:
          data += "None, "        
        data += (str(steering) + ", " + str(throttle) + "\n")
        print(data)
        catalog.write(data)
        global_count += 1


        
# = snapshot ================================================================================================
# Description:
#     
#
# Requires:
#   identify_lane() 
#   imageData = 
#   global count = number representing the image id
# Notes:
#  
#  
# Other:
#    grayscale: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ===========================================================================================================
def snapshot(imageData, global_count):
 
    # enforce color image
    #img = cv2.cvtColor(imageData,cv2.COLOR_BGR2RGB)
    # pre-process the image
    # uncomment this to do preprocessing in real time
    #img = identify_lane(img)
    # Create the new folder
    #write the image to the file at path specified above
    
    img = imageData
    img_name = 'sample_{}.jpg'.format(str(global_count)) 
    print("Saving Image")
    cv2.imwrite(os.path.join(full_folder_path, img_name), img)

# = identify_lane ===========================================================================================
# Description:
#  
#
# Requires:
#   
# Notes:
#  
#  
# Improvement Notes:
#   
#
# ===========================================================================================================      
def identify_lane(frame):

    # remove the QCAR bumper from view
    height, width = frame.shape[:2]
    cropped_height = int(height*0.85)
    frame = frame[:cropped_height,:]

    # store image size
    height, width = frame.shape[:2]

    # crop the field of view
    #poly_shape = np.array([(0,height),(int(width*0.5),0),(int(width*0.5),0),(width,height),],dtype=np.int32)
    #cv2.fillPoly(mask,[poly_shape],(255))
    #cv2.fillPoly(mask,ellipse,(255))

    # return frame 
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

    #gray_scaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    return frame

def main():
  drive()

if __name__ == '__main__':
    main()
