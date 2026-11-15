# Rover test bench

Standalone hardware test for the RaspRover: drive it over Wi-Fi from a browser and watch its
camera, using only the wheel motors, the wheel encoders and the IMU of the ESP32 driver board.
No ROS, no overhead camera. Used to check the hardware and the motion accuracy before the
guidance stack drives the rover.

- Arrow keys (or the on-screen pad) to drive, `Shift` = fast, `Space`/`Esc` = stop.
- Gyro steering assist while driving by hand: keeps the rover straight on slopes and slippery
  ground (see below).
- Camera pan-tilt: hold `Z` `Q` `S` `D` (W A S D on QWERTY), `C` = center.
- Trajectory following: draw points on the map (or pick a shape) and the rover follows them,
  or send a list of points to `POST /path` (the entry point for the future planner).
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
its true pose in green. `--sim-drift 0.15` (or the "Sim drift" slider) makes it turn by itself
while it moves, like on a slope, to try the steering assist; the simulated camera view shows
where the pan-tilt points.

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

## Trajectory following

The rover follows a list of waypoints, on its own odometry (wheel encoders + gyro), with no
outside position yet. It is the hook for the autonomous mode: a planner (drone + rover map,
obstacles) will send the trajectory, and send a new one whenever it changes its mind.

```http
POST /path
{"points": [[1.0, 0.0], [1.0, 0.5]], "frame": "local", "speed": 0.2}
```

- `frame: "local"` (default): metres relative to the rover when the step starts, x forward,
  y left. `"odom"`: the odometry frame, the `x`/`y` given by `/state` (`odom`), which the planner
  will map to its own frame once the overhead camera gives the rover's absolute pose.
- A new `POST /path` replaces the running trajectory; `POST /stop` stops it. A path is also a
  step in a sequence: `{"steps": [{"turn": 90}, {"path": [[...]]}, {"move": 0.1}]}`.
- `/state` → `ctrl.path`: the path being followed (odometry frame), the point aimed at, the
  distance done, and the distance to the path now and at most (`cte`, `max_cte`). The results
  table gives, once stopped, the length, the largest distance to the path and the end miss.

How: **pure pursuit**. The path is resampled every 2 cm; 20 times a second the rover finds the
closest path point ahead of its last progress, aims at the point `path_lookahead` further on
and drives the arc through it. It slows down before the end and before **corners** (waypoints
where the path bends more than `path_corner_deg`), stops on them and turns on the spot, and
it first turns to face the path if it starts the wrong way.

In simulation (with a 0.1 rad/s slope drift): 1 to 3 cm from the path on straight lines,
corners, a square and a circle, 3 cm on an S curve; the end point is reached within 2 to 3 cm.

| Setting | Default | Effect |
|---|---|---|
| `path_speed` | 0.2 m/s | Cruise speed (a path can ask for another one) |
| `path_lookahead` | 0.2 m | Shorter: follows curves more tightly, but may wobble |
| `path_lookahead_time` | 1.0 s | The look-ahead grows with speed: max(`path_lookahead`, time × v) |
| `path_corner_deg` | 45° | Bends sharper than this are corners: stop and pivot (0 = cut them) |
| `path_pivot_deg`, `path_pivot_exit_deg` | 60°, 10° | Turn on the spot when the path is this far off the nose |
| `path_w_max` | 1.5 rad/s | Sharpest rotation while following |
| `path_goal_tol` | 0.02 m | The end point counts as reached within this distance |

Not yet: obstacles (a planner or the depth camera will have to replace or stop the path), and
the odometry drifts over long runs (no absolute position until the overhead camera).

## Local guard (depth camera) and paths that take over

Off by default (`local_guard` 0), and only tried on the bench so far: see
[docs/local-navigation.md](../docs/local-navigation.md) before turning it on.

With `local_guard` 1, while a path is followed the controller looks at the obstacle grid of
the depth camera (`local_map.py`, fed from the capture server's `GET /grid`) along the path
ahead: it slows down when an obstacle is within `guard_slow_dist` and ends the path with
status `blocked` and `"obstacle": [x, y]` when one is within `guard_stop_dist`. Without a
recent grid it drives at `guard_blind_speed` at most. `GET /local_grid` gives the grid;
`/state` shows `local` and `ctrl.local`. Manual driving and `/move` are not guarded.

A `POST /path` with an `"id"` that replaces a running path takes over at the current speed,
from the closest point of the new path, instead of stopping and pivoting. The id comes back in
`ctrl.current` and in the history.

Deploy `local_map.py` with `rover_core.py` and `rover_server.py`.

## Tilt safety (anti-tipping)

Roll (sideways, + = left side up) and pitch (nose, + = nose up) come from a complementary
filter: the gyro (gx, gy) for fast changes, the accelerometer for the long term (trusted much
less during bumps). Their rates and angular accelerations are watched too.

- Between `*_warn_deg` and `*_stop_deg` the rover slows down (down to `tilt_min_scale`).
- At `*_stop_deg`, or when it is **tipping** (tilted more than `tip_min_deg` and the angle grows
  faster than `tip_rate_dps`, or accelerates away from level faster than `tip_acc_dps2`), it
  stops whatever it does (manual, move, path), backs off `tilt_backoff_dist` along the way it
  came, then refuses to go further that way until the angle is back under the warn angle.
- **The thresholds are placeholders** (15° warn, 25° stop): set them from the tipping tests.
- The back-off is a motion like the others: `/stop`, a lost ESP32 link or stale feedback
  (0.5 s) end it, and it never starts without fresh wheel feedback. A step ended by the guard
  finishes with the status `tilt: ...` (also in `ctrl.fault`), not an operator stop.

Tipping tests: turn the "Tilt guard" switch off, press "● Record", tilt or drive the rover until
it tips, stop the recording and download the CSV (20 Hz: commands, wheel speeds, roll, pitch,
rates, angular accelerations, raw IMU). The page also shows the peaks since the last reset.
The page warns if the gyro axes turn the wrong way compared with the accelerometer (fix with
`imu_roll_sign` / `imu_pitch_sign`). In simulation, the "Sim slope" sliders tilt the rover
(past 40° it falls over).

## Gyro steering assist (manual driving)

A software differential: the server adds a small rotation command to yours, which the ESP32
turns into a left/right wheel speed difference.

- **Straight input** (↑ or ↓ alone): the rover keeps the heading it had when you started going
  straight (or `assist_settle` s after a turn ended, once the rotation has died out). A PI
  correction on the heading error, with a dead band.
- **Turn input**: the rotation measured by the gyro follows the one you ask for (the rover skids
  in turns and turns slower than asked without it).
- It runs on the RDK, in the 20 Hz control loop: the ESP32 receives the same commands as before.
- Only rotations show on the gyro: a sideways slide on a slope, without rotation, is not
  corrected.

Settings to tune together on the real rover (Settings panel, saved on the robot):

| Setting | Default | Effect |
|---|---|---|
| `assist` | 1 | 1 on, 0 off (also the switch under the drive pad) |
| `assist_hz` | 20 | Corrections per second (at most 20, the loop rate) |
| `assist_deadband_deg` | 1.0 | Heading errors below this are ignored (avoids reacting to gyro noise) |
| `assist_k` | 3.0 | Heading hold: rad/s of correction per rad of error. Too high: the rover wiggles |
| `assist_ki` | 1.0 | Heading hold: integral, removes a steady pull (slope). Too high: slow swaying |
| `assist_k_rate`, `assist_ki_rate` | 0.6, 1.5 | Turns: how hard the rotation speed is corrected |
| `assist_w_max` | 0.8 rad/s | Largest correction |
| `assist_settle` | 0.3 s | Wait after a turn before holding the new heading |

In simulation with a 0.15 rad/s pull, 4 s straight end at 34° off course without the assist
and 2° with it.

## Camera pan-tilt

UGV firmware command `{"T":133,"X":pan,"Y":tilt,"SPD":..,"ACC":..}`: pan -180 (left) to 180
(right), tilt -30 (down) to 90 (up), bus servos ID 2 (pan) and 1 (tilt). The firmware does not
report the angles, so the page shows the commanded ones. The server sends at most `gimbal_hz`
(10) commands per second, only when the angles change, and centers the camera when the ESP32
connects. Settings: `gimbal_speed` (deg/s while a key is held), limits `pan_min`..`tilt_max`,
`pan_sign`/`tilt_sign` (-1 if an axis goes the wrong way), `gimbal_spd`/`gimbal_acc` (firmware
servo speed and acceleration, 0 = fastest). API: `POST /gimbal` (see `rover_server.py`).

Checked on our rover (2026-11-15): T:133 moves both servos although the boot mission sets
`{"T":900,"main":1,"module":0}` (no module). Pan +45 turns the camera right, tilt +30 up.

## Safety

- The ESP32 firmware stops the motors when it receives nothing for 3 s; the server sends a
  command every 50 ms.
- Manual driving stops 0.5 s after the last key press reaches the rover (Wi-Fi loss, closed tab).
- A move is abandoned after 20 s, or if the wheels do not turn for 1.5 s (rover blocked).
- `STOP`, `Space` or `Esc` stops everything, including queued steps.
- No wheel feedback (`T:1001`) from the ESP32 for 0.5 s (`Controller.FEEDBACK_TIMEOUT`), or the
  USB link drops: the rover stops, the step ends with status `no ESP32 feedback` or
  `ESP32 link lost`, the queue is emptied, and nothing restarts when the link comes back.
  The reason is in `/state` → `ctrl.fault`.
- A manual command (arrow key, `/cmd`, even `{v: 0, omega: 0}`) cancels the current step and
  the queued ones.
- An error in the 20 Hz loop stops the rover (status `controller error: ...`, counted in
  `/state` → `loop_errors`, `last_loop_error`); the loop keeps running, so `STOP` works. A bad
  ESP32 message is counted in `link.handler_errors` and the receive thread goes on.
- Numbers sent to the API must be finite and within limits (`MAX_*` in `rover_core.py`: 10 m
  per move, 720° per turn, path points within ±50 m, at most 1000 points and 50 m, manual
  speed 1 m/s): otherwise HTTP 400 and nothing moves. Steps before "Gyro OK": HTTP 409.
- `/config` checks every value (`CONFIG_LIMITS`: `manual_timeout` 0.1-2 s, so the dead man
  cannot be switched off; `track` and `path_speed` above 0...) and applies all or nothing; the
  file is written atomically, and a broken file is moved to `.bad` instead of stopping the
  server.
- `systemctl stop` / `kill` (SIGTERM) stops the rover like Ctrl-C.
- `/snapshot.jpg` never serves an old picture as new: header `X-Age-Ms`, and HTTP 503 when no
  picture newer than 2 s arrives within 3 s. `/state` has `boot_id` (new at each server start)
  and `uptime_s`.
- `/state` → `stop_count` counts the stops received since the server started (`POST /stop` and
  `/cmd {"T":0}`), also those pressed while the rover stands still, which leave no trace in
  `ctrl.history`. A client that drives the rover step by step (the mission,
  `merging_map/rover_reel.py`) reads it before each step: if it grew by more than the stops the
  client sent itself, someone pressed STOP, and the client must not send the next step. It
  starts again at 0 with a new `boot_id`; an older server has no such field.
- `POST /reset_odom` and `/calibrate_gyro` first stop whatever runs (step, queue, manual
  driving; logged `odometry reset` / `gyro recalibration`): a step in the `odom` frame would
  otherwise chase its old absolute target from the new origin. They do not count in
  `stop_count` (a client sees the reset by `odom.odometer` going down).
- A stop, a manual command, another client's step or a fault that comes during the 0.4 s
  settle after a step is logged with its own reason in `ctrl.history`, not as `done`.

## Deploying to the robot

The robot's own code as deployed on 2026-11-15 at 11:15 UTC (tilt guard, recorder, `/mode`)
is merged here with the safety changes above. The copy on the robot (`~/rover` on the RDK,
started by the `polaris-rover-api` unit with `--fps 40`) may have moved on since. Before
copying:

```bash
scp -r sunrise@<RDK IP>:~/rover /tmp/rover-robot      # keep the robot's version
diff -ru /tmp/rover-robot rover | less                 # merge what the robot has and we lack
```

Do not lose `rover_config.json` on the robot (the tuned settings): it is not in git. Then
copy, restart the unit and check `/state` (`boot_id` must be present).

To try by hand after this change, **wheels off the ground first**, then on the floor:

1. Gyro: right after the start, "Turn left 90°" must be refused (409, toast) until "Gyro OK".
2. Feedback watchdog: during "Forward 100 cm", unplug the ESP32 USB cable: the step must end
   `ESP32 link lost` (or `no ESP32 feedback`) at once; plug it back: the rover must NOT move by
   itself (the ESP32 reboots, ~7 s). Also check that normal moves and paths never end with
   `no ESP32 feedback` on their own (it would mean the feedback has gaps > 0.5 s: look at
   `link.rx_age` in `/state`).
3. Manual take-over: start `F50 L90 F50`, press an arrow during the first move, release: the
   rover must stay still (the turn and the second move are dropped).
4. Settings: "Save settings" from the page must still work (it sends every field); a value
   out of range must be refused with a message and change nothing.
5. `sudo systemctl restart polaris-rover-api` while driving slowly by hand: the rover must
   stop at once (log line "Stopping the rover.").
6. `curl -sD - http://<RDK IP>:8080/snapshot.jpg -o /dev/null` after 10 s without viewer:
   200 with a small `X-Age-Ms` (it waits for the camera), or 503 if the camera is broken.
7. Tilt guard: drive slowly onto a slope steeper than `pitch_stop_deg` (or lift the nose by
   hand while driving): the rover stops and backs off; press STOP during the back-off: it must
   stop at once. A step ended this way shows `tilt: ...` in the history and in `ctrl.fault`.

## Not handled yet

Obstacles (the depth camera is not used here), ROS topics, and the pose drift over long runs
(no absolute reference: the overhead camera will provide it).
