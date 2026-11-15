# Shared simulation: RaspRover + drone (MuJoCo)

A furnished apartment (a forest is coming) with the RaspRover and a drone, for testing the
drone, coordination and rover code without hardware. Plain Python, no ROS needed. Tested on
macOS so far (MuJoCo itself supports Linux and Windows; headless Linux may need
`MUJOCO_GL=egl` or `osmesa` for off-screen rendering).

What it gives you:

* the **rover camera** (RGB and exact depth) at the real mount: about 21 cm above the floor;
* a simulated **Intel RealSense D405** on the rover (colour + depth with a noise model, 7 cm to
  1 m). Its mount point (9 cm ahead of the body centre, 6 cm above the floor) and its noise model
  are assumptions until checked on the real robot; change the mount with `--d405-mount X Y Z`;
* a **drone camera** looking down, with its intrinsics and pose;
* the **true poses** of both;
* two **true occupancy maps**: what is seen from above, and what blocks the rover at its height
  (2-30 cm). They differ under tables and chairs: the rover fits under a table but stops at its
  legs. Nothing in the drone's view tells you that;
* rover commands in the **same JSON as the real RaspRover**: `{"T":1,"L":<m/s>,"R":<m/s>}`
  (per a Waveshare tutorial, not yet checked on our rover).

The scene uses CC0 assets from Poly Haven and the official Waveshare RaspRover meshes.

## Setup

In a virtualenv on your laptop (not in the ROS container):

```bash
pip install -r sim/requirements.txt      # mujoco, numpy, opencv-python
python sim/fetch_scene.py                # scene (~13 MB) + rover meshes, into sim/scenes, sim/assets
```

## Run it as if it were the real robots

```bash
python sim/run_world.py --auto           # the rover drives a planned route on its own
python sim/run_world.py                  # or drive it yourself through POST /cmd
```

HTTP on port 5050 (5000 is taken by AirPlay on macOS):

| Endpoint | What |
|---|---|
| `GET /video_feed` | rover camera, MJPEG (same shape as the RaspRover's own `/video_feed`) |
| `GET /drone_feed` | drone camera (looking down), MJPEG |
| `GET /state` | JSON: true rover pose, drone camera and rover camera intrinsics `K` and camera-to-world poses (OpenCV axes), grid resolution/origin, sim time |
| `GET /grid/overhead.png`, `GET /grid/rover.png` | true occupancy (white = blocked), image x right / y up |
| `GET /snapshot` | `.npz` of one simulation step: rover camera (`rgb`, exact `depth`, `K`, `T`), D405 (`d405_rgb`, `d405_depth` with noise, `d405_depth_true`, `d405_K`, `d405_T`), true pose `rover` = x, y, yaw |
| `POST /cmd` | `{"T":1,"L":0.2,"R":0.2}` (wheel speeds, m/s) or `{"v":0.2,"omega":0.3}`; 1 s watchdog |
| `POST /drone` | `{"x":..,"y":..,"z":..,"yaw":..}` |

TCP 7410 (`--state-port`, `0` to turn it off) streams the true rover state in the `guidance_rover_vo` wire format
(`src/guidance_rover_vo/guidance_rover_vo/protocol.py`). The bridge node listens to 7400 (the
rover mapper) by default; set its `port` parameter to 7410 to feed it the simulator's truth.

A pixel `(u, v)` of a camera with intrinsics `K` and camera-to-world pose `T` corresponds to world
point `p` by `[u, v, 1] ~ K @ T[:3, :3].T @ (p - T[:3, 3])`. `RoverWorld.project()` does it for you.

## Or use it directly from Python

```python
import sys; sys.path.insert(0, "sim")
from world import RoverWorld

w = RoverWorld("sim/scenes/apartment")
w.step(v=0.2, omega=0.0)               # one 0.1 s step; returns False if it would hit something
rgb, depth, K, T = w.rover_camera()    # what the rover sees, exact depth (m), intrinsics, pose
img, Kd, Td = w.drone_camera()         # what the drone sees
d = w.d405_camera()                    # dict: rgb, depth (noisy), depth_true, K, T
w.set_drone(x, y, z, yaw)
x, y, yaw = w.rover_pose               # ground truth
blocked = w.grid("rover")              # bool [i_x, j_y], cell 2 cm, origin at the room's min corner
ranges = w.ranges()                    # distance to obstacles at rover height, 24 sectors, front 180°
```

## Frames and units

World = `map`: metres, z up, origin at the room centre, floor at z = 0. Camera poses use OpenCV
axes (x right, y down, z forward). Rover heading 0 = +x.
