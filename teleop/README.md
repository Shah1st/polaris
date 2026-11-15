# teleop — drive the real rover and watch its 3D map build

A small web dashboard for the real RaspRover: ZQSD driving and the rover camera on the left, the
live 3D map of `rover_mapper/` on the right. It runs on the computer that reaches the rover's
server (`rover/rover_server.py`), for example a cloud machine behind an SSH tunnel, and is
viewed in a browser, itself often through another tunnel. Plain Python (`requests`, OpenCV,
numpy, scipy for the calibration), no ROS.

```
rover (rover/rover_server.py)  <--HTTP-->  teleop/server.py  <--browser-->  you
                                             |  /video_rect (lens removed)
                                             v
                                rover_mapper/run.py  -> 3D view on its own port
```

## Run

```bash
# 1. the dashboard (rover API reachable at --robot)
python teleop/server.py --robot http://127.0.0.1:18080 --port 8082

# 2. the mapper, fed with the rectified camera (see rover_mapper/README.md for its environment)
cd rover_mapper && python run.py --source http://127.0.0.1:8082/video_rect \
    --cam-height 0.22 --res 504 --max-jump 0.3 --max-depth 4 \
    --render points --point-stride 5 --live-hz 2 --port 8081 --no-record
```

Open `http://localhost:8082` (the 3D view is taken from port 8081 of the same host, so forward
both). `http://localhost:8082/sim` shows the forest simulation and the live map fusion of
`sim/fusion_live.py` when those run (`--sim`, `--fusion`).

`--cam-height` is the height of the lens above the floor (0.22 m on our rover): it sets the
scale of the whole map, so measure it. The rover must stand on the floor, not on a table.

## What the page does

* **ZQSD or arrows** (physical key positions, AZERTY or QWERTY), Space or Esc to stop, two speed
  sliders. Keys only work while the left panel has the focus; clicking in the 3D view releases
  them.
* **Direct link**: with `--robot-lan http://<rover LAN address>:8080`, a browser that is on the
  rover's own network talks to the rover directly (camera and commands), which avoids two trips
  across the Internet. Detected automatically, shown as "directe" / "via la VM".
* **Camera**: delivered one picture per request (`/frame.jpg`), a few requests kept open. A slow
  link then lowers the frame rate instead of piling up seconds of delay, which is what a pushed
  MJPEG stream did (measured: 12 s round trips). "économe" fetches about 4 pictures per second
  from the rover, "fluide" its whole stream; three picture sizes.
* **Figures**: battery, odometry, command delay, frames per second at each hop, browser <-> server
  round trip.

## Safety

The rover keeps its own protections (commands expire after 0.5 s on the rover, ESP32
heartbeat). On top of them:

* nothing is sent to the rover until a key is pressed;
* speeds are clamped in `server.py` (`V_MAX` 0.20 m/s, `W_MAX` 1.5 rad/s);
* a wanted speed expires 0.45 s after the last browser message (closed tab, lost focus, broken
  tunnel);
* a command that arrives more than 0.6 s later than the fastest one seen is dropped: a "go" that
  waited in a congested tunnel must not move the rover when it finally arrives. Stops are always
  obeyed;
* above a 1 s round trip the page refuses to drive through the server and says so.

Never start a second `rover_server.py` on the rover's serial port: opening it reboots the ESP32.

## The lens

The rover camera is a fisheye. The mapper's network assumes straight lines stay straight, so
`server.py` serves `/video_rect`, the picture with the lens removed, from the model in
`calib/plumbline.json`. That model was estimated from straight edges in six frames of one
viewpoint, without a calibration target:

```bash
python teleop/calibrate_lens.py --grab http://127.0.0.1:8082     # then look at calib/plumbline_check.jpg
```

It is an estimate: good enough to straighten the picture, less accurate than a checkerboard
calibration, and specific to this camera at 640x480.

## What was measured, and what was not

* Rover -> server link at the hackathon: about 0.5 s behind (frame by frame against fresh
  snapshots), more than 3 MB/s available.
* The 3D map of a first real drive had 8 pose jumps of 0.3 to 3.2 m (the mapper locking onto a
  lookalike place), which doubled the map. `--max-jump` refuses such poses; it was measured in
  the forest simulation (see `rover_mapper/`), not yet on a real drive.
* The pages were checked for syntax and by fetching every endpoint; the driving logic was tested
  against a mock rover (clamping, expiry, stale commands, stop). A real drive worked, but the
  latest version of the page has not been looked at in a browser by its author.
