# Rover test bench

Standalone hardware test for the RaspRover: drive it over Wi-Fi from a browser and watch its
camera, using only the wheel motors, the wheel encoders and the IMU of the ESP32 driver board.
No ROS, no overhead camera. Used to check the hardware and the motion accuracy before the
guidance stack drives the rover.

- Arrow keys (or the on-screen pad) to drive, `Shift` = fast, `Space`/`Esc` = stop.
- Measured moves: "forward 50 cm", "turn left 90°", or a sequence such as `F50 L90 F50 L90`.
- After each move or turn the rover stops and measures what it did. The page lists the
  result and the error, which is what the calibration below relies on.
- Telemetry: pose, heading from the gyro and from the wheels (their difference shows wheel
  slip), wheel speeds, battery, roll/pitch, CPU temperature, and a map of the path.

| File | Role |
|---|---|
| `rover_server.py` | Web server: page, MJPEG camera, HTTP API (see its docstring) |
| `rover_core.py` | ESP32 link, odometry (encoders + gyro), move/turn controller |
| `fake_esp32.py` | Simulated driver board, to try everything on a laptop |
| `web/index.html` | The control page |
| `tests/` | Controller tests against the simulated board (`pytest rover/tests`) |

`GET /video_feed` and `POST /cmd` (`{"v", "omega"}` or raw `{"T": ...}` commands) have the same
shape as in the shared simulation (`sim/run_world.py`, branch `Romain`), so code written against
the simulation can drive the real rover. Speed commands only last 0.5 s: repeat them at 5 Hz or more.

## Try it on a laptop (simulated rover)

```bash
pip install opencv-python-headless numpy   # only for the simulated camera view
python rover/rover_server.py --sim
```

Open http://localhost:8080. The simulated rover has the real one's defects (wheels slip in
turns, ground distance 2 % shorter than the encoders say, gyro bias), and the page also shows
its true pose in green.

## Run it on the rover (RDK X5)

Wiring: the RDK has its own 5 V supply and talks to the ESP32 board over a **USB cable** (on
our robot: `/dev/ttyACM0`, CH343 chip). Do not plug the RDK onto the robot's 40-pin header:
the two 5 V supplies fight each other.

The board runs Waveshare's **UGV** firmware (checked on 2026-11-14): `T:1001` feedback comes on
its own at 20 Hz with the wheel speeds (m/s, in 0.06 m/s steps), the raw IMU (gyro: 16.4 per
°/s), wheel counters `odl`/`odr` (cm) and the battery in centivolts. **Opening the USB port
reboots the ESP32** (about 7 s, it calibrates its IMU): the server waits for "UGV started".

```bash
# once, on the RDK (already done on ours)
sudo apt install python3-serial python3-opencv
sudo usermod -aG dialout sunrise    # serial port access, then log out and back in

# from the laptop, each time the code changes
scp -r rover sunrise@<RDK IP>:~

# on the RDK
python3 ~/rover/rover_server.py     # --port-serial /dev/ttyACM0 to choose the port
```

Keep the rover **still** after the start until the page shows "Gyro OK" (ESP32 boot, then
2 s of gyro bias), then drive. The page is at `http://<RDK IP>:8080`.

`calibrate_gyro.py` measures the gyro scale by turning the rover by hand (full turns).
Settings changed on the page are saved in `rover/rover_config.json` on the RDK (not committed).

Options: `--port 8080`, `--camera 0`, `--width 640 --height 480 --fps 15 --quality 70`,
`--no-camera`. The camera only runs while a page shows it (the RDK X5 has no fan).

## Calibration (Settings panel)

1. **Heading sign**: with the rover idle, turn it left by hand. "Heading gyro" must increase.
   If it decreases, set `yaw_sign` to `-1`.
2. **Distance**: run "Forward 100 cm" and measure with a tape. Set
   `dist_scale = real distance / distance measured`. Repeat 3 times: if the error changes a
   lot between runs, the wheels slip (floor, speed, `accel`).
3. **Turns**: run "Turn left 90°" a few times. If the rover does not pivot, raise `w_min`. If
   it often needs corrections ("Fix" column), adjust `lag` (higher if it overshoots).
4. **Square**: `F50 L90 F50 L90 F50 L90 F50 L90` must end close to the start mark on the floor.

## Safety

- The ESP32 firmware stops the motors when it receives nothing for 3 s; the server sends a
  command every 50 ms.
- Manual driving stops 0.5 s after the last key press reaches the rover (Wi-Fi loss, closed tab).
- A move is abandoned after 20 s, or if the wheels do not turn for 1.5 s (rover blocked).
- `STOP`, `Space` or `Esc` stops everything, including queued steps.

## Not handled yet

Obstacles (the depth camera is not used here), ROS topics, and the pose drift over long runs
(no absolute reference: the overhead camera will provide it).
