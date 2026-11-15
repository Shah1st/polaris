# Overhead camera pipeline

Status on 2026-11-15 (live camera reader, scale per image size, marker height, bad-frame
check). This file is the reference for the overhead
(top, drone) camera part of Polaris: what it does, what it outputs, what has been checked, and
what comes next. It is written for the team and for coding agents working on the repo. When the
pipeline or its outputs change, update this file in the same commit.

## What it does

From one image taken straight down, it produces a grid map of the obstacles and the rover's
position on that map. It needs no markers on the floor, and no trained model. The rover is
found by its size, or, reliably and with its heading, by an ArUco marker on its top (use the
marker for anything live: see "Live camera procedure").

## Steps

| # | Step | Where |
|---|---|---|
| 1 | **Scale.** Measured with a ruler (`metres_per_pixel`, recommended), or from the camera height and the lens for the actual image width. The map origin is the image's bottom-left corner, x to the right, y up. | `guidance_calib/ground_plane.py`, `calib_node.py`, `guidance_global_map/measure_scale.py` |
| 2 | **Top-down view.** The image is resampled to 5 mm per pixel. | `ground_plane.warp_to_top_down` |
| 3 | **Obstacles.** The floor colour is estimated around every pixel (so uneven lighting is followed); pixels that differ from it are obstacles. Small specks are dropped. | `guidance_global_map/segmentation.segment_obstacles` |
| 4 | **Rover.** With `rover_marker_id`, the ArUco marker on its top, corrected for its height (`marker_height`). Without, obstacle pixels are grouped into blobs and the blob whose size matches the rover is the rover. | `guidance_rover_pose/pose.py`, `segmentation.find_blobs`, `find_rover_blob` |
| 5 | **Grid.** The area is cut into 5 cm cells. A cell that an obstacle touches anywhere is an obstacle cell. | `segmentation.build_cell_grid` |

`guidance_global_map/pipeline.py` (`process_image`) runs all five on one image.
`guidance_global_map/camera_source.py` (`camera_reader`) supplies live images, and
`guidance_global_map/drone.py` (`OverheadDrone`) turns them into what `merging_map` asks of a
drone, rejecting bad frames.

The logic is in plain Python modules with no ROS import. The ROS nodes
(`calib_node.py`, `global_map_node.py`) only subscribe, call those modules and publish.

## Outputs

ROS topics, with the default configuration:

| Topic | Type | Contents |
|---|---|---|
| `/map` | `nav_msgs/OccupancyGrid` | The grid, frame `map`, 5 cm cells, republished about once a second |
| `/rover/pose_overhead` | `geometry_msgs/PoseWithCovarianceStamped` | Rover position in metres. The heading is not known: its variance is set to 1e6. |
| `/overhead/image_to_map` | `std_msgs/Float64MultiArray` | 3x3 pixel-to-metre mapping, row-major |
| `/overhead/debug/top_down`, `/overhead/debug/mask` | `sensor_msgs/Image` | The top-down view and the raw obstacle mask |

Grid cell values:

| Value | Meaning |
|---|---|
| 0 | Free, seen by the camera |
| 100 | Obstacle |
| 50 | The rover itself. Not a standard occupancy value: a planner must treat these cells as its start. |
| 99 | Margin around obstacles. Off by default (`margin: 0.0`). |
| -1 | Not seen by the camera |

Row 0 of the grid is the south edge (y = 0), column 0 the west edge (x = 0).

Without ROS, `python -m guidance_global_map.segment_image photo.jpg out/scene` runs the same
steps on one photo and writes the grid as a picture (`_grid.png`), as numbers (`_grid.csv`,
`_grid.npy`), in the map-fusion convention (`_fusion.npy`), and an overlay on the photo
(`_overlay.jpg`).

## Feeding the map fusion (`merging_map`)

`merging_map` merges drone and rover maps, plans the path and drives the rover. It is plain
Python with no ROS. It takes the drone's map as tiles, and its rover controller
(`rover_reel.RaspRoverReel`) has slots for the drone side. `guidance_global_map/drone.py`
fills them from this pipeline, also in plain Python:

| What `merging_map` wants | What `OverheadDrone` gives |
|---|---|
| Drone tiles at the start (first survey) | `tiles()` |
| `drone.disponible()` | True while the camera gives images |
| `drone.zoomer(tile)` | Takes a new image and returns the tiles overlapping the requested one |
| `position_drone()` -> rover (x, y, heading) or None | `rover_pose`; heading None without a marker (the rover adapter then measures it from the rover's motion) |
| "The picture after the rover stopped" (`rafraichir_drone`) | `refresh(force=True)`: with `camera_reader`, waits up to 2 s for a frame taken after the call |
| Planning area | `zone()`, to pass as `zone=` so paths stay inside the camera's view |

A real mission, with today's API (`merging_map/lancer_overhead.py` does the wiring):

```python
# PYTHONPATH=merging_map:src/guidance_calib:src/guidance_global_map:src/guidance_rover_pose
from lancer_overhead import lancer
from rover_reel import client_http
from guidance_global_map.drone import OverheadDrone, camera_reader
from guidance_global_map.pipeline import OverheadSettings

read = camera_reader('http://192.168.1.20:8080/video')  # or 0 (USB camera), rtsp://..., a file
camera = OverheadDrone(
    read,
    OverheadSettings(
        metres_per_pixel=0.0021,  # measured with measure_scale, for this stream's frame size
        camera_height=2.5,        # floor to lens; used for the marker's parallax
        rover_marker_id=10,       # ArUco DICT_4X4_50 id 10 on the rover's top
        marker_height=0.18,       # floor to the marker's surface, measured
    ),
)
try:
    lancer(camera, client_http('http://<rover>:8080'), victime=(1.8, 1.2))
finally:
    read.close()
```

`lancer` takes the start from `camera.rover_pose()` (or `depart=(x, y)`), refuses a start or a
victim outside `camera.zone()`, and plans with 5 cm cells and a 0.155 m robot radius.

How the grid meets the tile format:

| Tile format | This pipeline |
|---|---|
| Values 1 obstacle, 0 free and seen, -1 not seen | Converted from 100 / 0 / -1; the rover's own cells (50) are sent as free |
| 5 cm cells at the finest level (z = 4), 32 x 32 cells per tile | `cell_size: 0.05`, cut by `drone.cut_tiles` |
| Row 0 = south, tiles aligned to the global grid | Same; the grid starts at (0, 0) |
| Unseen area sent as -1, never as free | Cells outside the image are -1 |
| No margin from the sender (the planner grows obstacles itself) | `margin: 0.0`; keep it off |

Checked by hand against the `mergingmap` code as of commit `3cd2056`, in a scratch folder,
since that code is on another branch: `cut_tiles` gives exactly what their `decouper()` gives
(photo IMG_4099 and 200 random matrices); their map accepts the tiles from `tiles()` and
`zoomer()`; their planner returns a decision with `zone=camera.zone()`. `RaspRoverReel` and
`mission` themselves were not run with the adapter.

To import the adapter from their plain-Python side, put `src/guidance_calib`,
`src/guidance_global_map` and `src/guidance_rover_pose` on `PYTHONPATH` (or source the built
workspace) and have `opencv-contrib-python` and `numpy` installed.

### How the two are connected today

| Way | Entry point | What it does |
|---|---|---|
| Plain Python | `merging_map/lancer_overhead.py` | Runs a whole mission: `OverheadDrone` as the drone, the planner, and the real rover through its server (`rover/rover_server.py`). `--camera <stream>` for a live camera (read by `camera_reader`, with `--metres-par-pixel`), `--dry-run` first (no command reaches the rover), `--image <file>` for trials without a camera (only with `--dry-run`). `--suivi` writes the file the mission page shows, `--perception` adds the rover's own camera. |
| ROS | `merging_map_node` in `guidance_supervisor`, started by `guidance.launch.py` | Subscribes to `/map` and `/rover/pose_overhead`, waits for a goal on `/goal_pose`, publishes `/fusion/map`, `/fusion/plan` and `/fusion/decision`. It plans; it does not drive. |

Points that were open and how they were settled:

- **Rover heading.** The rover controller accepts a position with or without a heading. With
  no marker it drives a short straight line first and works out its heading from how the
  camera saw it move. That line is limited to the floor the map shows free all around the
  rover; next to an obstacle it is the shortest measurable step (12 cm), protected by bump
  detection only. It is sent as a `/path` to an absolute point of the odometry, so sending it
  again after a lost answer does not drive it twice; the point is 2 cm past the length wanted,
  since the rover server ends a path up to 2 cm early.
- **Planner settings for a small area**: 5 cm planning cells, arrival within 15 cm, and a
  rover radius of 0.155 m, the circle around the whole 0.20 x 0.23 m rover. The 0.115 m once
  proposed here was the half-width and let a corner touch an obstacle while turning.
- **Planning area.** Both ways plan only where the camera saw the floor: `camera.zone()` in
  the script, the cells of `/map` that are not unknown in the ROS node.
- **Live camera.** Two readers exist. `camera_reader(source)` (`camera_source.py`: MJPEG or
  snapshot URL, RTSP, USB index or file) is the one `lancer_overhead.py --camera` uses: it
  gives None when the newest frame is older than 2 s, and `read_new()` waits for a frame
  taken after the call. `drone.LiveCamera(source)` reads a stream with `cv2.VideoCapture`
  and always gives its newest picture. A stream has its own scale: measure
  `metres_per_pixel` on a picture of that stream (`measure_scale`, and
  `segment_image --metres-per-pixel` shows the result). After the rover stops, the script
  looks at a frame that arrived at least `--latence` seconds (0.5 by default) later, because a
  stream is late.

Still open:

- **The rover's own map.** `rover_mapper/perception_service.py` turns one rover camera image
  into a -1/0/1 grid around the rover, and `merging_map/perception_rover.py` gives it to the
  fusion (`lancer_overhead.py --perception URL`). Measured in simulation only: it has never
  seen the real rover's camera. `sim/fusion_feed.py` does the same for a depth image with a
  known camera pose; the D405 (`camera-test/`) is not mounted.
- **The frame** holds only while the camera does not move.
- ROS: `/map` and `/rover/pose_overhead` are published by the nodes; the plain-Python path
  above does not need them.

## Setting it up

Plain Python: the fields of `OverheadSettings` (`pipeline.py`). ROS: the same names in
`src/guidance_bringup/config/overhead.yaml`.

- `metres_per_pixel`: ground size of one image pixel, measured with a ruler (below). The
  recommended way for any live camera. It is per pixel of the frames the mission gets: measure
  again when the stream's resolution changes.
- Without it, `camera_height`, `focal_length_35mm` and `sensor_size` (`sensor_width_px`,
  `sensor_height_px`) give the scale: one sensor pixel covers a fixed angle, so the scale is
  the height times a constant of the camera mode (0.000330 per metre for the 12 MP photo
  mode of an iPhone 12 Pro Max, the defaults). What an image cannot say is whether it was
  cropped or resized, so the rule is fixed (`ground_plane.metres_per_pixel_for_view`):
  - the sensor's shape (4:3, either way up) with other pixel counts: the whole view resized,
    the scale follows the width;
  - any other shape: a crop, at the sensor's own scale. Right for a cropped photo; wrong for
    a video mode, which crops and resizes, hence one warning in the log.
- **Where the camera height goes.** `--hauteur` of `lancer_overhead.py`, `--camera-height` of
  `segment_image`, and `camera_height` in `overhead.yaml` (three nodes). The default is 1.8 m
  everywhere. It only has to be roughly right when one of the two checks below measures the
  scale. The dashboard's real-rover maps (`sim/fusion_reelle.py`) take no height: they do not
  use the overhead camera's map yet.
- **Scale read off the rover's marker** (`marker_size`, `--taille-marqueur`,
  `segment_image --marker-size`): with the side of the marker's black square given, the scale
  is its real side over its side in pixels, with no ruler and no exact height
  (`pipeline.marker_scale`). `OverheadDrone` keeps the value while the marker is hidden. A
  measured `metres_per_pixel` wins over it. The marker's side is read to about a pixel, so a
  marker 30 pixels wide gives the scale to about 3 %: make the marker as large as the rover's
  top allows.
- **Another phone or lens** changes `focal_length_35mm` (and `sensor_size` if the photo size
  differs). The 0.5x ultrawide lens of an iPhone 14 is 13 mm equivalent, still 4032 x 3024:
  `--focale-35mm 13` for `lancer_overhead.py`, `--focal-length-35mm 13` for `segment_image`,
  `focal_length_35mm: 13.0` in `overhead.yaml`. Each pixel then covers twice the ground
  (0.000660 per metre of height; 1.25 mm at 1.9 m, a view of about 5.1 x 3.8 m). Not tried on
  a real ultrawide photo. Its lens bends straight lines towards the edges of the picture, so
  the scale is less even across the view than with the main lens; keep "Lens Correction" on
  in the phone's camera settings, and keep the rover and the obstacles away from the edges.
- **The mission checks the scale on the rover** (`lancer_overhead.etalonner_echelle`,
  `scale_check.py`). Before anything else the rover drives 15 cm and back. Its wheels say how
  far it went; the camera sees how far the one thing that moved went; the ratio corrects the
  scale. With only a height given (it may be approximate), the measured scale replaces the
  derived one and is printed as a `--metres-par-pixel` value to reuse. With a measured scale
  given, the mission stops if the rover contradicts it by more than 10 %. The rover's size
  alone could not do this: at a scale 1.55 times too large a 13 x 15 cm pack measures
  20 x 23 cm, the rover's size, and was taken for it on IMG_4099. The rover needs 15 cm free
  ahead and behind; `--sans-etalonnage` skips the check. One move measures the scale to
  about 5 %; a ruler on the floor is still better.
- `rover_marker_id` and `marker_height`: the ArUco marker on the rover and its height above
  the floor. The marker is seen higher than the floor, so the floor mapping places it too far
  from the point under the camera, by d * h / (H - h) at a distance d from that point
  (H camera height, h marker height): 9 cm at 1 m with H = 1.9 m, h = 0.15 m. With
  `marker_height` set, the position is moved back towards the point under the camera by
  (H - h) / H; the heading does not change. **Default 0.0 = no correction.** Measure h with a
  ruler from the floor to the marker's printed surface, rover standing on the floor, and set
  `camera_height` (floor to lens) even when `metres_per_pixel` is measured. Assumes the camera
  looks straight down (the point under it is the image centre).
- `rover_size`: the rover as it appears from above (0.20 x 0.23 m measured on photos; the
  chassis is 0.17 x 0.18). Only used without a marker.
- `color_threshold`, `floor_window`: obstacle detection. `floor_window` must be wider than
  twice the largest obstacle.
- `cell_size`, `margin`: the grid.
- `OverheadDrone(..., bad_frame_hold=2.0, fresh_timeout=2.0, lost_rover_radius=0.30)`: see
  "Bad frames and a lost rover".

A marker setup on the floor also exists (`calib_node` `mode: markers`) for a tilted camera.
It only exists in the ROS node, which has never been launched; the plain-Python path assumes a
camera looking straight down.

## Live camera procedure

1. **Camera.** Fixed above the arena, looking straight down, high enough to see the whole
   arena (a phone main lens, 26 mm equivalent, sees about 1.33 x 1.00 m per metre of height:
   2.4 m of arena on the short side needs about 2.5 m). Main lens only, no ultrawide (no
   distortion correction). Lock exposure and focus if the app allows it.
2. **Source.** `camera_reader(source)` takes:
   - `0`, `1`, ... a USB camera on the computer running the mission;
   - `http://PHONE:8080/video` an MJPEG stream (phone apps such as "IP Webcam"), or a
     single-JPEG URL such as `http://PHONE:8080/shot.jpg`, which is polled;
   - `rtsp://...` a network camera;
   - a file path: a still image, re-read at each call (no motion: not for a mission).
   A thread keeps only the newest frame; `read()` returns None when it is older than
   `max_age` (2 s), so a frozen stream makes the camera unavailable instead of stale.
   Check: `python -m guidance_global_map.measure_scale SOURCE --distance 1 --save frame.jpg`
   prints the frame size and saves a frame.
3. **Scale with a ruler.** Lay a tape measure flat on the floor in the middle of the view,
   take two marks at least 1 m apart, and give their pixels:
   `python -m guidance_global_map.measure_scale frame.jpg --points U1 V1 U2 V2 --distance 1.0`
   (or `--click` on a laptop with a display). Use the printed `metres_per_pixel`. The view
   size it prints must match the arena.
4. **Marker on the rover.** Print id 10 of `DICT_4X4_50` (`python -m
   guidance_calib.generate_markers --ids 10 --out markers`), at least 10 cm wide with its white border, glue it flat
   on the rover's top, its top edge facing the rover's front (else set `yaw_offset`). Measure
   its height above the floor (`marker_height`). It gives the heading from the first frame
   and avoids every confusion of size detection.
5. **Arena.** A plain matte floor (a tarp or a carpet of one colour), obstacles of saturated
   colours (red, green, blue boxes; not white, grey or the floor's colour), nothing else in
   view: no tape lines, no cables, no bags, no people. Even, diffuse light with no hard
   shadows (a shadow of x0.75 counts as an obstacle). Keep obstacles narrower than half of
   `floor_window` (1 m).
6. **Check before driving.** Run `segment_image` on a saved frame with the same settings and
   look at the overlay: every obstacle red, the rover blue, nothing else red. Then start the
   mission (snippet above), with someone at the rover's stop button.

## Bad frames and a lost rover

- A black, white, overexposed, too dark or nearly uniform frame, or one whose contrast falls
  below a third of the last good frame's (hand in front of the lens), is not mapped. For
  `bad_frame_hold` seconds (2 s) `OverheadDrone` keeps the last good map, with its original
  time, and reports no rover pose (it may be out of date); after that the camera is
  unavailable (`disponible()` False, no tiles) until a good frame comes. A lasting change of
  contrast (over 10 s, frames otherwise fine) becomes the new reference. `drone.problem` says
  why the last frame was rejected. The ROS node `global_map_node` skips black, white and
  uniform frames.
- Without a marker, the rover is the blob of its size. When it is not found (touching a box,
  in its own shadow, seen larger far from the point under the camera) its blob would be an
  obstacle under itself. Instead, the tiles leave the cells within `lost_rover_radius`
  (0.30 m) of its last known position unseen (-1, so the fusion keeps what it knew), and the
  pose is None. This is a safety net, not tracking: a box of the rover's size can still be
  taken for the rover. **Use the ArUco marker for a live run.**

## What has been verified

- Unit tests of the logic on synthetic images and masks, and node start-up tests, run in
  the dev container (ROS 2 Jazzy, OpenCV 4.6) with the rest of the repository.
- Unit tests of the logic in plain Python on the VM, on 2026-11-15, among them:
  `camera_reader` against local HTTP servers (an MJPEG stream with and without
  Content-Length, a snapshot URL polled, a stalled stream, an unreachable URL), a video file
  through the OpenCV capture loop, a forced refresh using a frame made after the call;
  `LiveCamera` against a fake capture; the scale of a resized frame; the marker's parallax
  against a pinhole projection; black, white, grey, dark-noise and covered frames; a rover
  touching a box.
- The three ROS nodes were run once on the VM with stub ROS modules (new parameters,
  parallax, frame rejection), not with ROS: `rclpy` is not installed there.
- Six real photos (IMG_4097 to IMG_4102, floor mock-up, phone at 1.9 m) through the photo
  tool: all obstacles detected, the rover found in all six.
- One of them (IMG_4099) through the launched ROS nodes in the container, planner
  included: a goal sent on `/goal_pose` gave a path on `/fusion/plan` that stays inside the
  photo. Also `/map` and
  `/rover/pose_overhead` published, and the grid equal to the photo tool's except 1 of 1,188
  cells (the container has OpenCV 4.6, the photo tool was run with 5.0).

Not verified: any live camera or real stream; the scale against a ruler; the marker on the
real rover and its height; the bad-frame thresholds on a real camera (a very plain, evenly lit
floor with nothing on it may read as "nearly uniform"); a moving camera; the marker setup
through the launched nodes; anything outdoors.

## Decisions and why

| Decision | Why |
|---|---|
| Scale from camera height, not floor markers | There are no markers in the real world. Markers stay as a test rig. |
| Rover found by its size | No marker on the rover; no model to train. |
| Floor colour estimated locally | One colour for the whole image failed on unevenly lit photos. |
| Detect at 5 mm, publish at 5 cm | At 1 cm detection the drink can was missed. 5 cm is the finest tile level of the map fusion. |
| "Touched cell is an obstacle" | Asked for by the team; errs on the safe side. |
| Margin off | The map fusion grows obstacles by the robot radius itself; doing it twice closes real gaps. |
| Rover cells marked 50 on the map | Asked for by the team, so the map shows the rover. |

## Known limits

- No rover heading without a marker (the rover adapter measures it from the motion).
- Without a marker the rover is recognised by size alone: an obstacle of the same size, the
  rover touching an obstacle, its shadow, or parallax far from the point under the camera
  break it.
- With a marker, a disc of `rover_radius` (0.20 m) around it is never an obstacle: an
  obstacle closer than that to the rover is partly erased.
- The camera must look straight down, and the ground must be flat. The marker's parallax
  correction assumes the same.
- Tall obstacles lean away from the point under the camera and look larger than they are.
- Shadows count as obstacles. An obstacle the colour of the floor is missed. A person in view
  is an obstacle.
- No lens distortion correction: main lens only, not the ultrawide.

## Next steps, in order

1. A first run on the real rover with the live camera and the marker (procedure above):
   check the stream's scale and delay, and the planner's margin.
2. The rover's own map on the real rover (see "Still open" above).
3. Check the scale with a tape measure on the floor.
4. Make rover detection sturdier than size alone (colour, or tracking from the last
   position); mask the rover's rectangle instead of a 0.20 m disc.
5. A reference image of the empty arena, to see grey/white obstacles and ignore floor marks.
6. Later: tilted or moving camera without markers, uneven ground, a learned segmentation for
   outdoor scenes.

## For coding agents

- Put logic in the plain modules (`ground_plane.py`, `segmentation.py`, `pipeline.py`,
  `drone.py`, `pose.py`) with a pytest test. Keep the nodes to subscribing, calling and
  publishing.
- The grid values, the cell size, the frame and the row order are interfaces other parts
  depend on. Do not change them without updating this file and telling the team.
- Check a change on real photos with `segment_image`, and look at the overlay, before calling
  it done. The unit tests use synthetic scenes and did not catch the lighting problem.
- Tests: in the dev container, `colcon build --symlink-install && colcon test &&
  colcon test-result --verbose`. Outside it, the logic tests run with plain pytest once
  `opencv-contrib-python` and `numpy` are installed and `src/guidance_calib`,
  `src/guidance_global_map` and `src/guidance_rover_pose` are on `PYTHONPATH`; the node tests
  need ROS.
- `ruff check . && ruff format --check .` must pass.
- Say plainly what was run and what was not. The ROS nodes were untested for most of this
  work because the container could not be built; that was reported each time.
