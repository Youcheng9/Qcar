# V2V

Lets two QCars tell each other what they are doing, and slow down based on what the other car says.

## How it works

Each car sends a small message ten times a second over Wi-Fi (UDP, port 38900). The message says its throttle, steering, whether it is braking, whether it sees an obstacle, and which turn signal is on. Each car also listens for the other car's messages.

If the other car is braking, this car stops. If the other car sees an obstacle, this car drives at half speed. If no message has arrived for half a second, the other car is ignored and this car just keeps lane-following on its own.

## Files

- `v2v.py` - the link itself. No QCar needed, runs on any computer.
- `v2v_demo.py` - prints what a car hears from the other one. No driving.
- `../Demos/V2VDrive.py` - lane following plus V2V.

## Before you start

- Both cars and your laptop must be on the same Wi-Fi network.
- Give each car a different name with `--id` (for example `qcar1` and `qcar2`).
- If messages don't get through, use `--peer <other car's IP>` instead of broadcast.

## Try it

1. Check the link first, no driving. Run one on each car:

       python3 v2v_demo.py --id qcar1 --brake-after 8
       python3 v2v_demo.py --id qcar2

   qcar2 should print `clear`, then `peer-braking` after about 8 seconds.

2. Wheels off the ground, run on both cars:

       py3 V2VDrive.py --id qcar1
       py3 V2VDrive.py --id qcar2

   Press Ctrl+C on one car. The other should react.

3. Only then try it on the track at low throttle.

Add `--stop-on-link-loss` to `V2VDrive.py` if you want the car to stop when the other car goes silent, instead of carrying on.

## Not done yet

- The car doesn't know how far away the other car is. Following at a safe gap needs the LiDAR.
- The turn signal is sent but nothing reacts to it yet.
