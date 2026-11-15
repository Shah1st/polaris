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

## The mission page (for the jury)

`http://localhost:8082/mission` shows the real mission live, in large French text readable on a
projector: the fused map (free / obstacle seen by the drone / obstacle seen by the rover /
unknown), the zone, the rover and its heading, the victim, the planned path, the trajectory so
far, the current decision and its reason, the last lines of the journal, the mission state, the
rover camera and, when given, the perception debug picture. It reads the document that
`merging_map/lancer_overhead.py --suivi FICHIER` writes after each decision or step:

```bash
python teleop/server.py --robot http://127.0.0.1:18080 --port 8082 \
    --suivi /path/to/suivi.json [--perception-debug http://127.0.0.1:<port>/debug.jpg]
```

The page asks for one document or one picture at a time, about once per second, and the map is
only sent again when the file changed (`/api/mission?v=`): it stays usable on a slow link. The
age of the document and of each picture is shown, in red when old. No external file (no CDN, no
web font).

To see it without robot or mission (stand-in rover and perception, a sample drive that loops):

```bash
python teleop/demo_mission.py --port 8150          # then open http://127.0.0.1:8150/mission
python teleop/demo_mission.py --port 8150 --fixe   # the fixed sample teleop/mission_exemple.json
```

`--cam-height` is the height of the lens above the floor (0.22 m on our rover): it sets the
scale of the whole map, so measure it. The rover must stand on the floor, not on a table.

## What the page does

* **ZQSD or arrows** (physical key positions, AZERTY or QWERTY), Space or Esc to stop, two speed
  sliders. Keys only work while the left panel has the focus; clicking in the 3D view releases
  them.
* **Direct link**: with `--robot-lan http://<rover LAN address>:8080`, a browser that is on the
  rover's own network talks to the rover directly (camera and commands), which avoids two trips
  across the Internet. The banner under the title says which path the commands take
  ("LIAISON DIRECTE" / "VIA LA VM"). "automatique" switches to direct after two answers of the
  rover in a row and back after three misses, never while a key is held; "directe" and "via la
  VM" fix the path.
* **Camera**: delivered one picture per request (`/frame.jpg`), a few requests kept open. A slow
  link then lowers the frame rate instead of piling up seconds of delay, which is what a pushed
  MJPEG stream did (measured: 12 s round trips). "économe" fetches about 4 pictures per second
  from the rover, "fluide" its whole stream; three picture sizes. The age of the picture is
  written over it, in red above 1 s. After a restart of `server.py` the page follows the new
  frame numbers by itself (`X-Boot`).
* **Figures**: battery, odometry, command delay, frames per second at each hop, browser <-> server
  round trip.

## Safety

The rover keeps its own protections (commands expire after 0.5 s on the rover, ESP32
heartbeat). On top of them:

* nothing is sent to the rover until a key is pressed;
* speeds are clamped in `server.py` (`V_MAX` 0.20 m/s, `W_MAX` 1.5 rad/s); a value that is not
  a finite number (NaN, infinity) or a body that is not a JSON object is refused with 400;
* a wanted speed expires 0.45 s after the last browser message (closed tab, lost focus, broken
  tunnel);
* a command that arrives more than 0.6 s later than the fastest one seen from the same browser
  is dropped: a "go" that waited in a congested tunnel must not move the rover when it finally
  arrives. The reference is per browser and relearnt after 5 s without commands, so another
  computer's clock cannot block driving;
* a stop latches: Space, Esc or the button stop the rover, and every command of a key press made
  before the stop is refused, even one that arrives after it. Release the keys and press again
  to drive (the page numbers its presses; key auto-repeat is not a new press). Stops are always
  obeyed, whatever their body. In direct mode the rover's own server has no such latch;
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
* `teleop/test_server.py` tests the driving logic against a stand-in rover (`mock_rover.py`,
  never the real one): clamping, refused NaN/infinity, expiry, late commands per browser, the
  stop latch, malformed requests, the mission document and its relays, MJPEG readers that leave;
  and it parses the pages' JavaScript with esprima when installed:
  `python -m pytest teleop/test_server.py -q -p no:cacheprovider` (needs requests and OpenCV).
* No browser has rendered `/mission`, nor the latest `/` and `/sim`: they were checked for
  syntax and by fetching every endpoint. A real drive worked with an earlier version of `/`.
* Readers of `/video_feed` and `/video_rect` get a line break every 2 s when no frame arrives,
  so a reader that left is noticed (before, each mapper reconnection left a thread behind while
  the robot was offline). MJPEG readers skip it.
