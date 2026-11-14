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

`merging_map/fusion_carte.py` merges drone and rover maps and plans the path. It is plain
Python with no ROS and takes tiles, not `/map`. Nothing connects the two yet. What it expects
from the drone side, and how this pipeline meets it:

| It expects | This pipeline | Match |
|---|---|---|
| Values 1 obstacle, 0 free and seen, -1 not seen | 100 / 0 / -1, plus 50 for the rover | Converted by `segmentation.grid_to_fusion` |
| Cells of 5 cm at the finest level (z = 4) | `cell_size: 0.05` | Yes |
| Row 0 = south, columns towards +x | Same | Yes |
| Cells aligned to the global grid | Grid starts at (0, 0) | Yes |
| Unseen area sent as -1, never as free | Cells outside the photo are -1 | Yes |
| 32 x 32 tiles as messages `{"source", "t", "z", "x", "y", "data"}` | A dense matrix | Cut with their `decouper()`, below |
| Rover position passed separately to `decider()` | `/rover/pose_overhead` | Yes, without heading |
| No margin from the sender (it grows obstacles by 15 cm itself) | `margin: 0.0` | Yes, keep it off |

```python
from fusion_carte import CarteFusion, decouper
from guidance_global_map.segmentation import grid_to_fusion

carte = CarteFusion()                       # 32 cells per tile, 5 cm at z = 4
for msg in decouper(grid_to_fusion(grid), 0.0, 0.0, 4, carte, 'drone', t=now):
    carte.recevoir(msg)
decision = carte.decider((rover_x, rover_y), (goal_x, goal_y))
```

Checked once by hand on photo IMG_4099 with the `mergingmap` code as of commit `c17375e`: the
grid became 4 tiles, and `decider()` returned `SUIVRE` with a path for three different goals.
One of the paths left the photo and crossed unseen ground: the fusion allows unknown cells at
three times the cost.

Open points with the `mergingmap` author:

- How the tiles travel from this pipeline to the fusion (same process, a file, a socket, a ROS
  topic). Not decided.
- The frame. The fusion wants one global frame for drone and rover tiles. Here it is the
  image's bottom-left corner, which only holds while the camera does not move.
- The rover must place its own tiles in that frame, which needs its heading. This pipeline
  does not give the heading yet.

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

- Unit tests of the logic on synthetic images and masks, and node start-up tests: 35 tests,
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

1. Agree with the `mergingmap` author on how tiles are sent and on the common frame, then
   write the sender.
2. Rover heading from its motion between frames. The rover needs it to place its own map.
3. Check the scale with a tape measure on the floor.
4. Live phone stream instead of saved photos, with its own scale numbers.
5. Make rover detection sturdier than size alone (colour, or tracking from the last position).
6. Later: tilted or moving camera without markers, uneven ground, a learned segmentation for
   outdoor scenes.

## For coding agents

- Put logic in the plain modules (`ground_plane.py`, `segmentation.py`, `pose.py`) with a
  pytest test. Keep the nodes to subscribing, calling and publishing.
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
