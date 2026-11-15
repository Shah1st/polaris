# air_ground — guiding the rover from a camera that moves

Experiment (branch `experiment`): a simpler chain for the air-ground link. The air camera (an
iPhone today, a drone later) knows its own pose, the way a real drone does, and sends it with
every picture. Everything is then measured in ONE frame fixed to the floor, whatever the
camera's movements: the rover, the obstacles, the goal. Plain Python (numpy, OpenCV), no ROS.

```
iPhone (CamStream, ARKit mode)  --/stream: JPEG + X-Pose-->  camstream.py   picture + pose
                                                             pose_camera.py pixels <-> metres
                                                             rover_marker.py the rover: x, y, heading
```

Not there yet: the obstacle grid that accumulates, the path sent to the rover, the rover's
D405 proximity beam, the panels in the rover dashboard.

## State

* The geometry is tested on a synthetic camera that moves, tilts (to about 30 degrees) and
  turns: the floor's top view is the same from every pose, and the rover's marker is found
  within 5 mm and 2 degrees (`pytest air_ground`).
* **Nothing has run on a real iPhone.** The app's ARKit mode compiles on GitHub's Mac (see
  `iphone-camera/README.md`); the conventions below are Apple's documented ones, the first
  real picture will tell whether an axis is flipped.

## The floor frame

Metres, x and y on the floor, z up, z = 0 on the floor. It is ARKit's world frame (gravity
aligned, fixed in the room while the phone moves) with z up:

    x_floor = x_arkit      y_floor = -z_arkit      z_floor = y_arkit - floor_y

Its origin is under the place where the phone was when the tracking started, and its x axis is
arbitrary but fixed. It starts again when the app restarts its tracking: `sid` changes then.
The rover's odometry frame is tied to it by the rover's marker (to do: `map` -> `odom`, as in
AGENTS.md).

## The contract: `X-Pose`

One line of JSON, in the headers of each picture of `/stream` and of `/snapshot.jpg`, and as
the body of `/pose`:

```json
{"t": 5012.3381, "sid": 418233, "w": 1280, "h": 960,
 "k": [951.2, 951.2, 639.5, 479.6],
 "T": [16 numbers], "track": "normal", "floor_y": -1.3712, "floor_area": 2.41}
```

| Field | Meaning |
|---|---|
| `t` | time of the picture on the phone's clock (s) |
| `sid` | changes at each start of the tracking: a new floor frame |
| `w`, `h` | size of that JPEG (the sensor's own landscape picture, not rotated) |
| `k` | `fx, fy, cx, cy` in pixels of that JPEG |
| `T` | camera-to-world transform, 4x4, row after row; ARKit conventions: world y up, camera x right, y up, looking along -z |
| `track` | `normal`, `limited:initializing`, `limited:motion`, `limited:features`, `limited:relocalizing`, or `none` |
| `floor_y` | height of the floor in the world (the lowest large horizontal plane ARKit found under the camera), or `null` while it has found none |
| `floor_area` | area of that plane (m2), a hint of how sure it is |

A point of the camera frame `(x, y, z)` is at the pixel `(fx * x / -z + cx, fy * -y / -z + cy)`.

## Use

```python
from camstream import PoseStream
from rover_marker import RoverFinder

stream = PoseStream('http://<iPhone>:8080')
finder = RoverFinder(marker_id=0, size=0.10, height=0.12)   # the marker on the rover's roof
frame = stream.latest(max_age=0.5)
if frame and frame.pose and frame.pose.floor_known:
    top = frame.pose.floor_view(frame.image, (-1, -1, 1, 1), 0.005)   # stable top view
    rover = finder(frame.image, frame.pose)                           # x, y, heading, or None
```

The marker is an ArUco 4x4 (`DICT_4X4_50`), flat on the rover's roof, with its white margin.
`height` is the roof's height above the floor: without it the rover is seen several
centimetres off as soon as it is not right under the camera. `RoverSeen.scale_error` compares
the marker's measured side with the printed one: far from 0 means the pose, the floor height
or `height` is wrong.

## Known limits

* What stands above the floor leans away from the point under the camera in the top view (a
  20 cm box seen 0.5 m off axis from 2 m up looks about 5 cm longer). The rover is corrected
  by its known height; obstacles are not yet.
* ARKit needs texture and light to track: a plain glossy floor, fast moves or the dark give
  `limited:*`. Its frame drifts slowly (centimetres over minutes).
* The floor is found after a few seconds of slow movement above it.
* In ARKit mode the app sends no LiDAR depth, and the 1x lens only.
