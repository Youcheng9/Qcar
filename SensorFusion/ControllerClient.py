# ===========================================================================================================
# File: ControllerClient.py
# ===========================================================================================================
# Description:
#   PC-side client for FusionCollectionServer.py (and DataCollectionServer.py / TestDriveServer.py).
#   Reads a Logitech wheel or an Xbox gamepad with pygame and sends each event to the car over UDP
#   (port 38822) in the payload.py format:  |Controller ID 4|Event 4|Event Dimension 8|Event Value 8|
#
# Run on the PC (controller plugged into the PC, same network as the car):
#   pip install pygame
#   python ControllerClient.py <car-ip>                  # auto-detects wheel vs gamepad
#   python ControllerClient.py <car-ip> --mode gamepad   # force Xbox gamepad mapping
#   python ControllerClient.py <car-ip> --debug          # print raw axis/button numbers
#
# Wheel mode forwards events unchanged (the server was written for the Logitech wheel).
# Gamepad mode translates the Xbox pad into the wheel events the server understands:
#   Left stick X  -> steering           RT -> throttle pedal         LT -> brake (= stop label)
#   A / B         -> gear up / down     X  -> neutral                Y  -> toggle LEDs
#   LB / RB       -> forward / reverse  Start or Guide -> save and quit on the car
# ===========================================================================================================
import argparse
import socket
import sys
import time

import pygame

PORT = 38822
AXIS, BUTTON = 1536, 1539            # pygame.JOYAXISMOTION / JOYBUTTONDOWN event codes the server expects
WHEEL, PEDAL, BRAKE = 0, 1, 2        # server axis numbers (Logitech wheel layout)
GEARS = [11, 12, 13, 14, 15, 16, 17]  # server shifter buttons: 11 = R, 12..17 = gears 1..6
NEUTRAL, FORWARD, REVERSE, LEDS, QUIT = 0, 4, 5, 6, 10

# Xbox pad layout in pygame 2 on Windows (check with --debug if yours differs)
PAD_STEER, PAD_LT, PAD_RT = 0, 4, 5
PAD_A, PAD_B, PAD_X, PAD_Y, PAD_LB, PAD_RB, PAD_START, PAD_GUIDE = 0, 1, 2, 3, 4, 5, 7, 10
DEADZONE = 0.08

parser = argparse.ArgumentParser()
parser.add_argument('car_ip', help="the QCar's IP address")
parser.add_argument('--port', type=int, default=PORT)
parser.add_argument('--mode', choices=('auto', 'wheel', 'gamepad'), default='auto')
parser.add_argument('--joystick', type=int, default=0, help='index if several controllers are connected')
parser.add_argument('--debug', action='store_true', help='print raw pygame events')
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
target = (args.car_ip, args.port)


def send(joy_id, event, dim, value=0.0):
    msg = str(joy_id).zfill(4) + str(event).zfill(4) + str(dim).zfill(8) + ('%.4f' % value).zfill(8)
    sock.sendto(msg.encode('utf-8'), target)


def main():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() <= args.joystick:
        sys.exit('No controller found. Plug it in (USB or Bluetooth) and try again.')
    joy = pygame.joystick.Joystick(args.joystick)
    joy.init()
    name = joy.get_name()
    mode = args.mode
    if mode == 'auto':
        mode = 'wheel' if any(k in name.lower() for k in ('wheel', 'g920', 'g29', 'g923', 'racing', 'driving force', 'logitech')) else 'gamepad'
    print('Controller: %s (%d axes, %d buttons) -> %s mode, sending to %s:%d'
          % (name, joy.get_numaxes(), joy.get_numbuttons(), mode, *target))

    gear = 1                          # index into GEARS; server starts at the slowest forward gear
    pending = {}                      # latest value per axis, sent at most every 10 ms
    last_flush = 0.0
    clock = pygame.time.Clock()

    try:
        while True:
            for e in pygame.event.get():
                if args.debug and e.type in (pygame.JOYAXISMOTION, pygame.JOYBUTTONDOWN):
                    print(pygame.event.event_name(e.type), e.dict)

                if e.type == pygame.JOYAXISMOTION:
                    if mode == 'wheel':
                        pending[e.axis] = e.value
                    elif e.axis == PAD_STEER:
                        pending[WHEEL] = 0.0 if abs(e.value) < DEADZONE else e.value
                    elif e.axis == PAD_RT:     # trigger: -1 released .. 1 pressed; pedal: 1 released .. -1 floored
                        pending[PEDAL] = -e.value
                    elif e.axis == PAD_LT:
                        pending[BRAKE] = -e.value

                elif e.type == pygame.JOYBUTTONDOWN:
                    if mode == 'wheel':
                        send(e.joy, BUTTON, e.button)
                        if e.button == QUIT:
                            return
                    elif e.button in (PAD_A, PAD_B):
                        gear = min(gear + 1, len(GEARS) - 1) if e.button == PAD_A else max(gear - 1, 0)
                        send(e.joy, BUTTON, GEARS[gear])
                        print('Gear', 'R' if gear == 0 else gear)
                    elif e.button == PAD_X:
                        send(e.joy, BUTTON, NEUTRAL)
                    elif e.button == PAD_Y:
                        send(e.joy, BUTTON, LEDS)
                    elif e.button == PAD_LB:
                        send(e.joy, BUTTON, FORWARD)
                    elif e.button == PAD_RB:
                        send(e.joy, BUTTON, REVERSE)
                    elif e.button in (PAD_START, PAD_GUIDE):
                        send(e.joy, BUTTON, QUIT)
                        print('Sent quit; the car is saving the session.')
                        return

                elif e.type == pygame.JOYDEVICEREMOVED:
                    print('Controller disconnected, stopping the car.')
                    return

            now = time.time()
            if pending and now - last_flush >= 0.01:
                for axis, value in pending.items():
                    send(joy.get_instance_id(), AXIS, axis, value)
                pending.clear()
                last_flush = now
            clock.tick(200)
    except KeyboardInterrupt:
        pass
    finally:
        # leave the car stopped: wheel straight, pedals released
        for axis, value in ((WHEEL, 0.0), (PEDAL, 1.0), (BRAKE, 1.0)):
            send(0, AXIS, axis, value)
        sock.close()
        pygame.quit()


if __name__ == '__main__':
    main()
