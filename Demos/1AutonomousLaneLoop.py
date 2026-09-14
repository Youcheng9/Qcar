'''
"""
=========================================================================================
QCar Autonomous Driving Program - Front Camera Classification Model Integration
=========================================================================================

Description:
------------
This program controls the Quanser QCar in both manual and autonomous driving modes. 
It utilizes a front-facing camera for data collection and a pre-trained classification model 
to predict steering directions. The model outputs one of 17 distinct steering classes, 
which are then mapped to specific steering angles. The program supports both user-controlled 
manual driving and AI-driven autonomous mode.

Key Features:
-------------
1. **Manual Mode**: The user can control the car using the keyboard inputs (W, A, S, D).
2. **Autonomous Mode**: The QCar collects real-time camera images, processes them through 
   a TensorFlow Lite (TFLite) model, and adjusts its steering based on model predictions.
3. **LED Indicators**: The QCar uses LED signals to indicate turns, braking, and headlights.
4. **Throttle and Steering Control**: The car moves forward or backward with throttle adjustments,
   while steering is adjusted based on manual inputs or model predictions.
5. **Camera Support**: The program only uses the front camera (Camera ID = 3) for autonomous 
   driving and ignores other camera feeds.
6. **Model Loading**: The program loads a TFLite model from the "Models/" directory. 
   The model file path should be specified on line 45.

How It Works:
-------------
1. **Initializing the QCar**:
   - The QCar object is created with `readMode=0`, allowing control over the car's motors and sensors.
   - The pygame library is initialized to handle user inputs and display a control interface.

2. **Model Loading**:
   - A TensorFlow Lite model is loaded via `loadModel("Models/Feb25.tflite")`.
   - The model expects an input image and returns an array of probabilities for 17 steering classes.

3. **Driving Modes**:
   - The program starts in manual mode.
   - The user can switch between manual and autonomous modes using the following keys:
     - Press **M** for manual driving.
     - Press **N** for autonomous mode.

4. **Manual Driving Mode**:
   - The user can control the QCar using the keyboard:
     - **W**: Move forward.
     - **S**: Decelerate/Brake.
     - **A**: Steer left.
     - **D**: Steer right.
     - **R**: Engage reverse mode.
     - **F**: Switch back to forward mode.
     - **H**: Toggle headlights.

5. **Autonomous Driving Mode**:
   - The program continuously captures images from the front camera.
   - The image is preprocessed (converted to grayscale and cropped).
   - The image is sent to the TFLite model for classification.
   - The predicted steering class is mapped to a steering value.
   - The throttle is locked at **0.085** to maintain forward motion.
   - Steering is adjusted based on the model’s prediction.

6. **Camera Functions**:
   - `camPreview(camIDs=["front"])` captures an image from the front-facing camera.
   - The image is converted into the correct format for the model.
   - Other cameras (left, right, back) are initialized but not used.

7. **Stopping the Car**:
   - The program stops if the user closes the pygame window.
   - Pressing **Q** resets the throttle and steering to 0.

How to Use:
-----------
1. Place the desired **TFLite model** inside the `Models/` folder.
2. Modify the model path on **line 45** if necessary.
3. Run the program using: py3 AutonomousLaneLoop.py


4. Start in manual mode and use the **M** key to control manually.
5. Press **N** to switch to autonomous mode.
6. To stop the program, close the pygame window or press **Ctrl + C** in the terminal.

Dependencies:
-------------
Ensure the following libraries are installed before running the program:
- pygame
- tensorflow
- numpy
- opencv-python
- pal-utilities (Quanser QCar library)

Limitations:
------------
- The program only supports **classification models**.
- It **does not** use depth cameras or other sensor inputs.
- The **front camera** is the sole data input for the model.
- Throttle is **fixed** in autonomous mode for safety.
- The program assumes the **model outputs a single class prediction**.

Authors:
--------
- **Alex Sanna** (ajsanna@cpp.edu)
- **Matthew Baldivino** (mabaldivino@cpp.edu)

Last Revised:
-------------
- **10/20/2024** - Initial Testing
- **02/25/2025** - Code Revision and Optimization

"""


'''

# import libraries 
import pygame
from pal.products.qcar import QCar
from pal.utilities.math import *
import numpy as np
import time
from pal.utilities.vision import Camera2D
import sys
import cv2
import numpy as np
import threading
import os
from pal.products.qcar import QCarCameras, IS_PHYSICAL_QCAR
from pal.utilities.vision import Camera2D
import tensorflow as tf
import os 


# create qcar object 
myCar = QCar(readMode=0)

# Main Program for driving the car. 
def Drive():
    print("Driving started.")
    pygame.init()

    '''
        This is where you specify what model you would like to use. TFLITE recommended. 
        Models should be stored in the Models/ folder in the Demos/ folder. Case Sens.
    '''
    #driving_model = loadModel("Models/Mar12Class.tflite")
    driving_model = loadModel("Models/qcar_steering_tf231.tflite")
    print("Model is loaded")
    input_details = driving_model.get_input_details()
    output_details = driving_model.get_output_details()
    
   
    # setup screen size for pygame display control panel. 
    screen_width = 800
    screen_height = 600
    # screen = pygame.display.set_mode((screen_width, screen_height))
    input_details = driving_model.get_input_details()
    print("got input details")

    output_details = driving_model.get_output_details()
    print("got output details")

    screen_width = 800
    screen_height = 600

    print("before set_mode")
    # screen = pygame.display.set_mode((screen_width, screen_height))
    print("after set_mode")

    # pygame.display.set_caption("QCar Autonomous Demo")
    # pygame.display.flip()

    #print("screen:", screen)
    print("display driver:", pygame.display.get_driver())
    print("pygame display initialized?", pygame.display.get_init())
    print("caption set")


    # CAR PARAMS: set up +- increment and maximum
    deceleration_increment = .00001
    throttle_increment = 0.005
    steering_increment = 0.005
    max_throttle = 0.1
    min_throttle = -0.1
    max_steering = 0.5
    min_steering = -0.5
    MINIMUM_THROTTLE_EPSILON = .0000001
    AUTONOMOUS_MODE = True

    # default speed and direction at stand still 
    throttle = 0.0
    steering = 0.0

    def stop():
        throttle = 0.0
        steering = 0.0
    
    # Default LED status
    LEDs = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    isReverse = False
    headlights_on = True
    s_key_pressed = False
    turn_signal_timer = 0.0
    turn_signal_interval = 0.5
    turn_signal_state = 0 

    counter=0

    # steering classes coorolated to model prediction via "One Hot Encoding"
    steering_classes = [-0.5, -.4375, -.375, -.3125, -.25, -.1875, -.125, -.0625, 0, .0625, .125, .1875, .25, .3125, .375, .4375, 0.5]
    #steering_classes = [-0.5, -.4375, -.375, -.125, -.0625, 0, .0625, .125, .375, .4375, 0.5]
    #steering_classes = [-0.5, -.4375, -.375, -0.25, -.125, -.0625, 0, .0625, .125, .25, .375, .4375, 0.5]

    running = True
    while running == True:
        if AUTONOMOUS_MODE is False: # start in user input mode
            #listen for pygame Events in the event queue
            for event in pygame.event.get():
                #if the window is closed, shut the car off
                if event.type == pygame.QUIT:
                    running = False
                    print("Shutting Down")
                    exit(0)
                #if finger has been taken off the keys
                elif event.type == pygame.KEYUP:
                    #maintain steering direction
                    if event.key == pygame.K_a or  event.key == pygame.K_d:
                        steering = 0.0
                    #if car is in drive and the throttle key is released
                    if event.key == pygame.K_w and isReverse is False:
                    #reduce throttle amount
                        while throttle > MINIMUM_THROTTLE_EPSILON:
                            throttle -= .0000001
                    #car should come to a complete stop if it has reached minimum throttle before the throttle key is pressed again
                    throttle = 0
                    #if car is in reverse and throttle key is released
                    if event.key == pygame.K_w and isReverse is True:
                    #reduce throttle amount
                        while throttle < MINIMUM_THROTTLE_EPSILON:
                            throttle += .0000001
                    #car comes to complete stop upon reaching minimum throttle before the throttle key has been pressed
                    throttle = 0

            keys = pygame.key.get_pressed()
            
            # steering
            if keys[pygame.K_w]:
                if not isReverse:
                    throttle += throttle_increment
                    #prevents car from going over max throttle amount
                    throttle = min(throttle, max_throttle)
                    LEDs[4] = 0
                    
                if isReverse:
                    #prevents car from going under min throttle amount unless it is manually set to 0 like in deceleration algorithm
                    throttle -= throttle_increment
                    throttle = max(throttle, min_throttle)

            elif keys[pygame.K_s]:
                if throttle > 0 and not isReverse:
                    throttle -= throttle_increment
                    #prevent car from going under min throttle amount
                    throttle = max(throttle, min_throttle)
                    LEDs[4] = 1
                    s_key_pressed = True  # Set flag when 'S' key is pressed
                elif throttle < 0 and isReverse:
                    throttle += throttle_increment
                    #prevent car from going under min throttle amount
                    throttle = max(throttle, min_throttle)
                else:
                    s_key_pressed = False  # Reset flag when 'S' key is released

            if not keys[pygame.K_s] and not s_key_pressed:
                LEDs[4] = 0

            if keys[pygame.K_r]:
                isReverse = True
                throttle = 0
                LEDs[4] = 0  # Turn off LED[4] when 'R' key is pressed
                LEDs[5] = 1
            elif keys[pygame.K_f]:
                isReverse = False
                throttle = 0
                LEDs[5] = 0

            if keys[pygame.K_d]:
                steering -= steering_increment
                #turning right is steering in the negative direction
                steering = max(steering, min_steering)
            elif keys[pygame.K_a]:
                steering += steering_increment
                #turning left is steering in the positive direction
                steering = min(steering, max_steering)
            
            if keys[pygame.K_q]:
                steering , throttle = 0.0 , 0.0
                LEDs[5] = 0
            
            if keys[pygame.K_h]:
                headlights_on= not headlights_on
                LEDs[6] = 1 if headlights_on else 0
                LEDs[7] = 1 if headlights_on else 0
            
            if keys[pygame.K_n]:
                print("Autonomous Mode Selected")
                print("Loading: " + str(driving_model))
                AUTONOMOUS_MODE = True
            
            # update turnled timer
            turn_signal_timer += 0.01

            # based on direction and speed update led control
            if steering > 0.15:
                if turn_signal_timer >= turn_signal_interval:
                    turn_signal_state = 1 - turn_signal_state
                    turn_signal_timer = 0.0
                if turn_signal_state:
                    LEDs[0] = 1
                    LEDs[2] = 1
                else:
                    LEDs[0] = 0
                    LEDs[2] = 0
            elif steering < -0.15:
                if turn_signal_timer >= turn_signal_interval:
                    turn_signal_state = 1 - turn_signal_state
                    turn_signal_timer = 0.0
                if turn_signal_state:
                    LEDs[1] = 1
                    LEDs[3] = 1
                else:
                    LEDs[1] = 0
                    LEDs[3] = 0
            else:
                LEDs[0] = 0
                LEDs[1] = 0
                LEDs[2] = 0
                LEDs[3] = 0

            if throttle < 0:
                LEDs[5] = 1
            else:
                LEDs[5] = 0
  
        elif AUTONOMOUS_MODE is True:
            for event in pygame.event.get():
                #if the window is closed, shut the program down.
                if event.type == pygame.QUIT:
                    running = False
                    print("Shutting Down")
                    exit(0)
            # to switch back to user mode press M         
            keys = pygame.key.get_pressed()
            if keys[pygame.K_m]:
                print("Manual Mode Selected.")
                print("Switching to keyboard control.")
                throttle = 0
                steering = 0
                AUTONOMOUS_MODE = False
            else:
                
                throttle = .075 # locked throttle for steer only models 
                image_cap_np = camPreview(camIDs=["front"]) #take image and store as NP array for efficiency 
                height, width = image_cap_np.shape[:2] 
                image_cap_np = image_cap_np[int(height*.5):int(height*.85),:]
                #crop image to ROI by slicing in half to where only the ground is visible.
                image_cap_grayscale = cv2.cvtColor(image_cap_np, cv2.COLOR_BGR2GRAY) # black and white
                image_cap_grayscale = image_cap_grayscale[np.newaxis,:,:, np.newaxis] #should be shape 1,77,420,1
                image_cap_grayscale = image_cap_grayscale.astype('float32') #model takes in data as float32.
                
                photo = image_cap_grayscale
                height, width = photo.shape[:2]

                offset = int(width * 0.05)  # Adjust this as needed

                # Define top-left and top-right triangular regions with a gap
                top_left_triangle = np.array([[0, 0], [width // 2 - offset, 0], [0, height // 1.5 - offset]], np.int32)
                top_right_triangle = np.array([[width, 0], [width // 2 + offset, 0], [width, height // 1.5 - offset]], np.int32)

                # Create a mask (white image).
                mask = np.ones_like(photo, dtype=np.uint8) * 255
                mask = mask.astype('float32')
                # Draw black triangles over the corners
                cv2.fillPoly(mask, [top_left_triangle, top_right_triangle], (0, 0, 0))
                
                # Apply the mask to remove the corners
                image_cap_grayscale = cv2.bitwise_and(photo, mask)
                #take 10 pixels off the top for further cropping down to 1,67,420,1
                #print(image_cap_grayscale.shape)
                image_cap_grayscale = image_cap_grayscale[:,10:,:,:]
               
                #send image to the model
                driving_model.set_tensor(input_details[0]['index'], image_cap_grayscale)
                driving_model.invoke()
                #predictions is the 17 index array of probabilities [x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x] where 0<=x<=1
                predictions = driving_model.get_tensor(output_details[0]['index'])
                counter +=1
                print(f"Counter: {counter}")

                class_assign = np.argmax(predictions) #find index of max probability using argmax.
                #print(class_assign)
                steering = steering_classes[class_assign] #give steer index to the car for immediate use. 

        # update qcar control
        #print("Throttle: " + str(throttle) + " __ Steering: " + str(steering))
        LEDs[6] = 1 
        LEDs[7] = 1 
        LEDs[4] = 1 
        myCar.read_write_std(throttle=throttle, steering=steering, LEDs = LEDs)


camera_right = Camera2D(cameraId="0",frameWidth=420,frameHeight=220,frameRate=30)
camera_back = Camera2D(cameraId="1",frameWidth=420,frameHeight=220,frameRate=30)
camera_left = Camera2D(cameraId="2",frameWidth=420,frameHeight=220,frameRate=30)
camera_front = Camera2D(cameraId="3",frameWidth=420,frameHeight=220,frameRate=30) 

def camPreview(camIDs, steering = 0, throttle = 0, catalog= None):
    if camIDs is not None:
        if "front" in camIDs:
            camera_front.read()
            if camera_front is not None:
                return camera_front.imageData
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

# Code below is used to load in the model file and initialize storage for its use.        
def loadModel(filename):
    driving_model = tf.lite.Interpreter(model_path=filename)
    driving_model.allocate_tensors()
    return driving_model
        
        
try:
    Drive()
except KeyboardInterrupt:
    print("Stopping QCar...")
    myCar.read_write_std(throttle=0.0, steering=0.0, LEDs=np.zeros(8))
    myCar.terminate()
