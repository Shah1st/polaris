# Overhead camera pipeline

Status on 2026-11-14, branch `overhead-camera`. This file is the reference for the overhead
(top, drone) camera part of Polaris: what it does, what it outputs, what has been checked, and
what comes next. It is written for the team and for coding agents working on the repo. When the
pipeline or its outputs change, update this file in the same commit.

## What it does

From one image taken straight down, it produces a grid map of the obstacles and the rover's
position on that map. It needs no markers on the floor or on the rover, and no trained model.

## Steps

| # | Step | Where |
|---|---|---|
| 1 | **Scale.** The camera height and the lens give the size of one pixel on the ground. The map origin is the image's bottom-left corner, x to the right, y up. | `guidance_calib/ground_plane.py`, `calib_node.py` |
| 2 | **Top-down view.** The image is resampled to 5 mm per pixel. | `ground_plane.warp_to_top_down` |
| 3 | **Obstacles.** The floor colour is estimated around every pixel (so uneven lighting is followed); pixels that differ from it are obstacles. Small specks are dropped. | `guidance_global_map/segmentation.segment_obstacles` |
| 4 | **Rover.** Obstacle pixels are grouped into blobs and measured in metres. The blob whose size matches the rover is the rover. | `segmentation.find_blobs`, `find_rover_blob` |
| 5 | **Grid.** The area is cut into 5 cm cells. A cell that an obstacle touches anywhere is an obstacle cell. | `segmentation.build_cell_grid` |

`guidance_global_map/pipeline.py` (`process_image`) runs all five on one image.

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

## Feeding the map fusion (`merging_map`, branch `mergingmap`)

`merging_map` merges drone and rover maps, plans the path and drives the rover. It is plain
Python with no ROS. It takes the drone's map as tiles, and its rover controller
(`rover_reel.RaspRoverReel`) has slots for the drone side. `guidance_global_map/drone.py`
fills them from this pipeline, also in plain Python:

| What `merging_map` wants | What `OverheadDrone` gives |
|---|---|
| Drone tiles at the start (first survey) | `tiles()` |
| `drone.disponible()` | True while the camera gives images |
| `drone.zoomer(tile)` | Takes a new image and returns the tiles overlapping the requested one |
| `position_drone()` -> rover (x, y, heading) or None | `position_drone`; needs the marker on the rover for the heading |
| Planning area | `zone()`, to pass as `zone=` so paths stay inside the camera's view |

```python
from cerveau import mission
from fusion_carte import CarteFusion
from rover_reel import RaspRoverReel, envoyeur_http
from guidance_global_map.drone import OverheadDrone, image_file_reader
from guidance_global_map.pipeline import OverheadSettings

camera = OverheadDrone(
    image_file_reader('overhead.jpg'),            # or any function returning a BGR image
    OverheadSettings(camera_height=1.9, rover_marker_id=10),
)
carte = CarteFusion()
for message in camera.tiles():
    carte.recevoir(message)
rover = RaspRoverReel(carte, envoyer=envoyeur_http(...), lire_odometrie=..., percevoir=...,
                      position_drone=camera.position_drone, drone=camera)
mission(rover, carte, depart=..., victime=..., zone=camera.zone())
```

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

### Needed from the `mergingmap` author

1. **Use the adapter**: feed `camera.tiles()` at the start, and pass `position_drone`, `drone`
   and `zone` as in the snippet above.
2. **Accept a rover position with or without a heading.** The rover has no marker for now, so
   this pipeline knows where the rover is but not which way it faces. `RaspRoverReel` counts
   the drone as available only when `position_drone()` returns a position, and it assigns the
   heading from it without checking. So today the adapter returns None, and the mission runs
   in "drone lost" mode the whole time: half speed, position from wheel odometry only and
   never corrected by the camera, and a return-and-wait once the uncertainty passes 60 cm.
   Both cases should work:
   - **No heading (now).** Take `(x, y, None)`, correct the position, and keep the odometry
     heading. `OverheadDrone.rover_pose()` already returns this.
   - **With a heading (later).** Take `(x, y, heading)` as today. It will come from an ArUco
     marker on the rover (id 10, `DICT_4X4_50`, top edge facing forward), which this pipeline
     already reads, or from the rover's motion between frames, which is not built.

   Once `RaspRoverReel` handles a None heading, `position_drone=camera.rover_pose` is the only
   change in the snippet above.
3. **Planner settings for our small area**, all arguments that exist already. On photo
   IMG_4099 the defaults block 45% of the floor (10 cm planning cells, 15 cm radius rounded up
   to 20 cm) and count a goal as reached 50 cm away. Proposed until a test drive gives real
   numbers: `CarteFusion(niveau_plan=4, rayon_robot=0.115)` for a 15 cm margin, and
   `tolerance=0.15`.
4. **The frame.** Start, goal and rover-camera tiles must be in this pipeline's frame: origin
   at the image's bottom-left corner, x right, y up, metres. It holds while the camera does
   not move.
5. **The demo test and matplotlib.** Without matplotlib installed, a plain `pytest` in the
   container collected nothing at all once `merging_map` was in `testpaths`. The Dockerfile
   on this branch now installs it; the image has to be rebuilt.
6. **ROS, since the team wants it.** The adapter above works without ROS. To join the ROS
   side, `merging_map` needs a node around it, for example in the placeholder
   `guidance_supervisor` package, that subscribes to `/map` and `/rover/pose_overhead` and
   calls the same module. `/map` becomes tiles with `segmentation.grid_to_fusion` and
   `drone.cut_tiles`. This pipeline's side of it is done: both topics are published. Who
   writes the node is not decided.

## Setting it up

All parameters are in `src/guidance_bringup/config/overhead.yaml`.

- `camera_height`, `focal_length_35mm`, `sensor_width_px`, `sensor_height_px`: the scale. The
  defaults are a photo from an iPhone 12 Pro Max main camera at 1.9 m. The image may be
  cropped, not resized. Video has a different scale than photos.
- `rover_size`: the rover as it appears from above (0.20 x 0.23 m measured on photos; the
  chassis is 0.17 x 0.18).
- `color_threshold`, `floor_window`: obstacle detection. `floor_window` must be wider than
  twice the largest obstacle.
- `cell_size`, `margin`: the grid.

A marker setup also exists (ArUco markers on the floor and on the rover; `mode: markers`). It
handles a tilted camera and gives the rover's heading. See the README.

## What has been verified

- Unit tests of the logic on synthetic images and masks, and node start-up tests: 45 tests,
  run in the dev container (ROS 2 Jazzy, OpenCV 4.6).
- Six real photos (IMG_4097 to IMG_4102, floor mock-up, phone at 1.9 m) through the photo
  tool: all obstacles detected, the rover found in all six.
- One of them (IMG_4099) through the launched ROS nodes in the container: `/map` and
  `/rover/pose_overhead` published, and the grid equal to the photo tool's except 1 of 1,188
  cells (the container has OpenCV 4.6, the photo tool was run with 5.0).

Not verified: the absolute scale against a ruler; a live camera stream; a moving camera; the
marker setup through the launched nodes; anything outdoors.

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

- No rover heading without a marker.
- The rover is recognised by size alone: an obstacle of the same size, or the rover touching
  an obstacle, breaks it.
- The camera must look straight down, and the ground must be flat.
- Tall obstacles lean away from the point under the camera and look larger than they are.
- Shadows count as obstacles. An obstacle the colour of the floor is missed. A person in view
  is an obstacle.
- No lens distortion correction: main lens only, not the ultrawide.

## Next steps, in order

1. Run a mission of `merging_map` with the adapter on the real rover, with a marker on the
   rover, once its author has done the points listed above.
2. Rover heading without a marker, from its motion between frames.
3. Check the scale with a tape measure on the floor.
4. Live phone stream instead of saved photos: a reader function for `OverheadDrone`, and
   the stream's own scale (`metres_per_pixel`, measured with a ruler in view).
5. Make rover detection sturdier than size alone (colour, or tracking from the last position).
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
