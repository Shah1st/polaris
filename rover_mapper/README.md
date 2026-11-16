# RoverMap — live 3D map + 3rd-person view of the RaspRover

The Mac reads the rover's camera stream, reconstructs the environment in 3D with
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) and displays
the robot moving through it, with a follow camera as on recent cars.
Nothing to install on the robot.

## Run

```bash
.venv/bin/python run.py --robot IP_DU_ROVER --cam-height 0.10
```

Then open http://localhost:8080.

- `--cam-height`: height of the lens above the floor, in metres. **To be measured**: it sets the scale of the whole map.
- `--source URL`: if the stream is not at `http://IP:5000/video_feed` (address assumed from the Waveshare documentation, not verified on the robot).
- `--model depth-anything/DA3-SMALL`: faster, less accurate. `--res 504`: slightly more accurate, slower.
- `--source 0` (the Mac's webcam), `--source 1` (iPhone through Continuity Camera, after granting camera permission in macOS) or `--source video.mp4 --loop` to test without the robot.

## Measurements (MacBook Pro M5, GPU, DA3-BASE, resolution 336)

TUM sequence `freiburg1_desk` (handheld camera, 9.4 m path), one frame in 4:

| Scenario | Trajectory error | Keyframes | Lost frames |
|---|---|---|---|
| Single pass | 3.8 cm | 80 | 0 |
| Out and back (return into the known area) | 3.6 cm on the return | +4 on the return | 0 |
| Abrupt jump to the starting point, 6 laps | 3.7 cm on each lap | +2 on the 2nd lap, then 0 | 1 per lap (then recovered) |
| 40 frames from another place, then return to the desk | 3.7 cm on the return | 0 during the other place, +2 on the return | 40/40 in the other place, 0 on the return |

- **Speed**: 0.155 s per frame, i.e. about 6 Hz. The display is animated at 30 Hz between two poses.
- **Consistency**: the depth of the map and the trajectory agree to within about 5 %.
- **Absolute scale**: the map calibrated from the floor is about 14 % too large (11 to 22 % depending on the settings) on this sequence. DA3's metric depth model is wrong in the other direction (about 8 % too small). Neither of the two is a reliable reference; on the rover, prefer the distance travelled by the wheels or a stereo camera.
- **Depth sensor (optional, not connected for now)**: `Mapper.step(rgb, sensor_depth)` accepts a depth in metres aligned with the image (e.g. RealSense D405), and the scale then comes from the sensor instead of the floor. Tested with the TUM Kinect cut at 0.8 m to imitate a short-range sensor: depth correct to within 5 % instead of about 14 % error, trajectory about 10 % too short, accuracy unchanged (3.7 cm). The rest comes from a scale drift of about 5 % along the path. This is not a trial with a real D405.
- **Background refinement** (`Mapper.refine_keyframe`, each keyframe recomputed with 4 neighbours at 504): keyframe positions 3.8 → 3.0 cm, but depth unchanged (7 % error). Gain judged too small to enable it; the function remains available.

The measurements are made with `scripts/eval_tum.py` (`mode=once|fwdrev|loop|gap`) and `scripts/eval_refine.py`.

## Live trials (iPhone, Continuity Camera)

Launcher: `scripts/live_iphone.command`, to be opened with the macOS Terminal app, which will ask for camera access. Each session is recorded in `out/rec_<date>.avi`, with a per-frame log in `out/live_log.csv`. To replay a session identically: `python scripts/replay.py out/rec_….avi [début_s] [fin_s]`.

- **People and moving objects**: a small segmentation model (DeepLabV3-MobileNetV3, 11 ms per frame on the Mac's GPU) spots people, animals and vehicles. These pixels are excluded from the map and from tracking. On a pass with people walking by: tracking losses 29 % → 9 %. Option `--no-mask` to disable it; it is also disabled automatically as long as the model weights are not downloaded.
- **Rendering**: the map is displayed as "Splats (gaussiennes)" (Gaussian splats), one disc per point oriented along the surface, or as "Points". The "Rendu" (rendering) menu of the view switches from one to the other, and `--render` chooses the rendering at startup.
- **"Reconnecter la caméra" (reconnect the camera) button**: reopens the camera and starts a new recording. The reconnection also happens by itself if a live stream stops.
- **Ignored frames**: black frames (lens covered, Continuity Camera paused) and frames without detail (blurred, overexposed) are not used and never become keyframes.
- **References**: only those that the current frame really overlaps count. A bad keyframe therefore no longer makes the whole tracking fail.
- **Measurements**: on a pass at floor level in a room (the rover's point of view), tracking is lost about 15 to 25 % of the time. These are mostly brief losses (0.1 to 0.5 s), caused by motion blur, people walking by or very close-up views. Filming a desk with one's hands moving in the field of view gives markedly more losses.

## Known limits

- No global loop closure: the map can drift over a very long path. Tracking does however reuse the nearest keyframes, which limits the drift when the robot passes through the same place again.
- When tracking is lost, the position is frozen (red indicator) and the program looks for a known place resembling the current frame. If the robot has carried on into an area that was never mapped, it stays frozen: you have to return to a known area or reset the map. An automatic recovery exists (`lost_patience`) but is disabled, because in testing it inserted a keyframe coming from an unknown place.
- Live, with the test video looping, the number of keyframes still grows slowly (less than one per lap, versus 0 with still images).
- Absolute scale correct only to about ±15 % on the sequence tested (see above).

## Files

- `rovermap/source.py`: reading of the stream (MJPEG, webcam or video), keeping only the most recent frame.
- `rovermap/mapper.py`: tracking, keyframes, floor detection and point cloud.
- `rovermap/viewer.py`: 3D scene (viser), rover model and follow camera.
- `scripts/fake_rover.py`: simulates the rover's stream from a video.
- `scripts/eval_tum.py`: accuracy measurement on a TUM dataset (`0` as the frame step = real-time pace).

## The rover's real D405 (`--d405`)

The robot only captures and sends (`d405_server.py` on the rover's computer, 127.0.0.1:8091, reached from the VM through an SSH tunnel): a JPEG colour image and the raw 16-bit depth in the frame of the depth sensor, not aligned, with factory intrinsics and extrinsics, only when it is asked for. All the computation is done on the VM (`rovermap/source.py`, `D405Source`): depth in metres (raw × `depth_scale` read from the device, the unit of the D405 is not the mm), depth → colour alignment for "D405 seule" (D405 only), depth → robot camera reprojection for "Les deux" (both).

```bash
# D405 only (scale given by the sensor)
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/run.py --d405 http://127.0.0.1:8091 \
    --sensors d405 --d405-height 0.06
# Both: image from the robot (rectified by the dashboard) + D405 depth
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/run.py --source http://127.0.0.1:8082/video_rect \
    --rgb-fov 100 --d405 http://127.0.0.1:8091 --sensors both --d405-mount TX TY TZ TANGAGE \
    --cam-height 0.21 [--rover http://127.0.0.1:18080]
```

- `--d405-height`: height of the D405's lens above the floor (to be measured).
- `--d405-mount TX TY TZ [TANGAGE [LACET [ROULIS]]]`: colour camera of the D405 seen from the robot camera, OpenCV axes (x to the right, y down, z forward, metres); pitch > 0 = the D405 looks lower, yaw > 0 = further left, roll > 0 = its right side lower. **To be measured.**
- `--rgb-fov`: horizontal field of view of `--source`, a centred pinhole image (the dashboard's `--rect-fov`, 100 by default). Not `--robot`: its raw stream is fisheye.
- `--d405-max-age 1.0`: an older D405 frame is not used (age measured by the robot's monotonic clock plus ours; the robot's wall clock is hours off).
- **"Les deux" (both): the two images are not captured together.** The robot camera arrives through the dashboard ~0.5 s late, the D405 is the more recent: when turning or driving fast, the depth lands on the wrong pixels. Drive slowly.
- **Pan-tilt**: the robot camera is on a pan-tilt; `--d405-mount` is valid only at pan 0 and at the tilt of the measurement. With `--rover URL` (the rover's server, `GET /state`), a non-zero pan is reported.

Without a camera connected, `d405_server.py` looks for it after 2 s, then 4, 8 and 15 s at most (each attempt enumerates USB: one attempt every 2 s took 23 % of one core of the robot); as soon as frames arrive, the wait goes back to 2 s. The current wait is in `/health` (`retry_s`).

Without the robot: `scripts/fake_d405.py` serves the same HTTP contract from the apartment simulator (`--sim http://127.0.0.1:8090`, 2 frames/s at most; never the forest one on 5050). Tests: `test_d405_source.py` (fake D405 server with non-trivial extrinsics, without device or model).

## Obstacle grid on the robot (`d405_server.py`, `GET /grid`)

For local navigation ([docs/local-navigation.md](../docs/local-navigation.md)): the capture
server itself computes, on the robot, the grid of 5 cm cells in front of the camera from
the measured depth (`rovermap/perception.py`, numpy and OpenCV only, without a model), 10
times per second at most and only when it is asked for. `POST /calibrate_floor`, with the rover
set on a flat, clear floor, measures the floor plane once and keeps it (`d405_floor.json`).
Height of the lens: `--cam-height 0.15`. The `rovermap/` folder must be next to the server
on the robot. Tried only with a fake camera (`test_d405_grid.py`).

## Rover perception for the mission (a single image, no tracking)

`perception_service.py` gives `merging_map` (interface 4: `merging_map/perception_rover.py`, `percevoir_http(url)`) the 5 cm cells in front of the rover: 1 obstacle, 0 floor seen, -1 unknown. On each request: latest image from the camera → DA3 on this single image → floor plane (RANSAC, lower half) → scale given by `--cam-height` → rover frame (x forward = optical axis projected onto the floor, y to the left, z from the floor) → obstacle = points between `--band-low` 4 cm and `--band-high` 40 cm above the floor; whatever is behind an obstacle or outside the field of view stays -1 (a 0 would erase the drone's obstacles). No tracking: the mapper's did not hold up on the real rover.

```bash
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/perception_service.py \
    --source http://127.0.0.1:8082/video_rect --cam-height 0.22 --fov 100 --port 8095
# GET /obstacles (JSON, 503 without an image less than 2 s old; ?fresh=1 waits for an image that arrived
# after the request), GET /debug.jpg (image with red/green cells), GET /health
```

`--fov 100`: `/video_rect` is by construction a centred pinhole image of 100° (the dashboard's `--rect-fov` option); without `--fov`/`--k`, the intrinsics estimated by DA3 are used. Bound to 127.0.0.1 only.

**Real rover on a slow link (recommended command).** Through the SSH tunnel, each request to the robot costs 1.3 to 4 s whatever its size: the service reads the robot's raw image directly, one request per look, without going through the dashboard (which pulls the stream continuously):

```bash
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/perception_service.py \
    --source http://127.0.0.1:18080/snapshot.jpg --calib teleop/calib/plumbline.json \
    --rect-fov 100 --fov 100 --cam-height 0.22 --range 1.5 --port 8095
```

- `--calib FICHIER`: model of the fisheye lens (`teleop/calib/plumbline.json`). The source then gives RAW images (the robot's `/snapshot.jpg`, fisheye 640x480) and the service rectifies each one itself, with the same K and the same `cv2.fisheye.initUndistortRectifyMap` tables as `CameraHub` in `teleop/server.py` (the test compares byte for byte with `/video_rect`). An image of a size other than that of the model is refused.
- `--rect-fov DEG` (100 by default): horizontal field of view of the rectified image; with `--calib` it is also the known field of view (`--fov`, if given, must be equal; `--k` is refused).
- `--fetch-timeout 10`: timeout of one GET of the image (the link takes 1.3 to 4 s; the old timeout of 3 s mistook a slow response for a failure).
- A URL that ends in `.jpg`, `.jpeg`, `.png` or `.npz` is a single image: no probe request at startup (on a slow link, this probe made the first request return 503).
- `GET /obstacles?fresh=1`: with such a source, ONE new GET of the image per request (fresh by construction); its duration is in `fetch_s` (the cost of the link, visible to the caller). Without `fresh=1`, an image less than `--max-age` s old is reused without a request, and `GET /debug.jpg` right afterwards shows the same image and the same result, without a new request.
- `POST /obstacles` (body: a JPEG, raw with `--calib`, already rectified otherwise): for a driver that has already downloaded the image; same JSON as `GET /obstacles`, with `age_s` 0, `t` = time of receipt and `fetch_s` null. The service then opens no connection to the robot. Unreadable or wrong-size image: 400.
- On the caller's side: a timeout of at least `--fetch-timeout` + the inference (~0.6 s) for `?fresh=1`; the pan of the pan-tilt is not taken into account here (the caller rotates the view, `cap - pan`).

**Measurements in simulation** (`scripts/eval_perception.py`, 30 images of the recorded forest + 30 of the apartment, same classification applied to the exact depth, range 2 m, cells to within 1 cell; inference 0.08-0.1 s on the shared L4, 0.57 s per request in the live service):

| | apartment, known K | apartment, DA3's K | forest, known K | forest, DA3's K |
|---|---|---|---|---|
| scale (depth / ground truth, 0-1 m) | 0.98 | 1.19 | 0.91 | 1.17 |
| obstacles 0.5-1 m: precision / recall | 0.71 / 0.77 | 0.67 / 0.61 | 0.35 / 0.59 | 0.47 / 0.80 |
| obstacles 1-1.5 m: precision / recall | 0.49 / 0.60 | 0.45 / 0.36 | 0.22 / 0.36 | 0.28 / 0.37 |
| "free" cells that are obstacles, 0.5-1 m / 1-1.5 m | 0.2 % / 1.9 % | 1.4 % / 3.8 % | 0.5 % / 1.8 % | 0.5 % / 1.7 % |

At the exact cell, precision drops to 0.46 (apartment, 0.5-1 m, known K): the positions are correct to within ~5 cm, no better. Beyond 1-1.5 m recall collapses: default range 1.5 m. Indoors, the known K is better (scale and obstacles); in the forest, DA3 with its own K is. **Never tried with the camera of the real rover.** To check first: the real height of the lens (it sets the scale), the pan-tilt being level (the floor plane absorbs a tilt, but the x axis follows the optical axis: a non-zero pan rotates the whole view), `/debug.jpg` in front of a box placed at 0.5 m and 1 m, and whether the front of the rover is in the image (`--min-range`).
