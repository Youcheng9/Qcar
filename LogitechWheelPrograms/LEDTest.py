import time
import numpy as np
from pal.products.qcar import QCar

car = QCar(readMode=0)

LEDs = np.zeros(8)

try:
    while True:
        for i in range(8):
            LEDs[:] = 0
            LEDs[i] = 1
            print("Turning on LED", i)
            car.write(throttle=0, steering=0, LEDs=LEDs)
            time.sleep(1.5)

finally:
    car.write(throttle=0, steering=0, LEDs=np.zeros(8))
    car.terminate()
