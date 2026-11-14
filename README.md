# Polaris: drone/rover guidance POC

Hackathon proof of concept for a guidance system. An overhead 2D camera (on a drone or a fixed
mount) sees the whole area and builds the global map, and the rover's position comes from the
same camera. A rover with a 3D depth camera follows the planned path and avoids local obstacles
on its own. Everything runs on one laptop with ROS 2 Jazzy.

**Status:** the overhead-camera nodes (`guidance_calib`, `guidance_global_map`,
`guidance_rover_pose`) have a working baseline, described in
[Overhead camera](#overhead-camera). The supervisor is still a placeholder, and Nav2,
robot_localization and the rover are not wired up yet. `TODO(...)` comments mark what is missing.

```
overhead camera ─┬─► guidance_global_map ─► /map ────────────────► Nav2 planner
                 ├─► guidance_rover_pose ─► robot_localization ─► Nav2 (rover pose)
                 └─► guidance_calib (image → ground plane, used by the two above)

rover depth camera ─► Nav2 local costmap ─► Nav2 controller ─► rover

guidance_supervisor: sends goals to Nav2, tracks mission state, watches health
```

## Packages

| Package | Build type | Purpose |
|---|---|---|
| `guidance_interfaces` | ament_cmake | Custom msg/srv/action definitions (empty for now; rosidl needs CMake) |
| `guidance_calib` | ament_python | Overhead camera calibration, image → ground plane mapping |
| `guidance_global_map` | ament_python | Overhead image → `nav_msgs/OccupancyGrid` on `/map` |
| `guidance_rover_pose` | ament_python | Rover pose in the map frame from the overhead camera |
| `guidance_supervisor` | ament_python | Goals to Nav2, mission state machine, health checks |
| `guidance_bringup` | ament_python | `guidance.launch.py` and `config/` for parameters |

Local obstacle avoidance comes from Nav2 (controller + local costmap fed by the depth camera),
set up in `guidance_bringup`, so it has no package of its own.

## Dev environment

The Docker image is based on the official `ros:jazzy` image and adds Nav2, robot_localization,
cv_bridge, image_publisher, OpenCV, pytest and ruff.

**VS Code:** open the folder and run *Dev Containers: Reopen in Container*. The first start runs
`rosdep install` and `colcon build`.

**Plain Docker:**

```bash
docker build -t polaris-dev .
docker run -it --rm -v "$PWD":/ws polaris-dev bash
```

Camera passthrough is not set up yet. On a Linux host add `--device=/dev/video0` (see the TODO in
`.devcontainer/devcontainer.json`). On Windows, USB cameras need `usbipd-win` to reach WSL2 first.

## Build and run

Inside the container, from `/ws`:

```bash
rosdep install --from-paths src --ignore-src -y -r
colcon build --symlink-install
source install/setup.bash

# all nodes
ros2 launch guidance_bringup guidance.launch.py
ros2 launch guidance_bringup guidance.launch.py log_level:=debug

# a single node
ros2 run guidance_global_map global_map_node
```

## Overhead camera

The overhead camera is fixed and looks straight down at the floor. Its image arrives on
`/overhead/image_raw`. No markers are needed, on the floor or on the rover.

| Node | Publishes | How |
|---|---|---|
| `calib_node` | `/overhead/image_to_map` (3x3 homography, row-major, latched) | Scale from the camera height and the lens. The map origin is the image's bottom-left corner, x right, y up. |
| `global_map_node` | `/map` (`nav_msgs/OccupancyGrid`), `/rover/pose_overhead` (`geometry_msgs/PoseWithCovarianceStamped`), debug images `/overhead/debug/top_down` and `/overhead/debug/mask` | Resamples the image to a metric view at 5 mm. Pixels whose colour differs from the floor around them are obstacles. The obstacle blob that has the rover's size is the rover: it is left out of the map and its position is published. The map is a grid of 2.5 cm cells; a cell that an obstacle touches anywhere is occupied. |
| `rover_pose_node` | nothing in this setup | Only used with a marker on the rover (see below). |

All parameters are in `src/guidance_bringup/config/overhead.yaml`. The ones to set for a new
setup:

- `camera_height`, and `focal_length_35mm` with `sensor_width_px` / `sensor_height_px` for the
  camera. The defaults are an iPhone 12 Pro Max main-camera photo. The image may be cropped but
  not resized, and video has a different scale than photos.
- `rover_size`: the rover as it appears from above, cables and shadow included. Measure it on a
  photo with the tool below. No other object in view may have the same size.
- `color_threshold` and `floor_window`: tune while watching `/overhead/debug/mask`.
- `cell_size`: the size of a map cell. `margin`: extra blocked cells around obstacles, off by
  default.

Map cell values: 0 free, 100 obstacle, 99 margin, -1 not seen by the camera.

**Trying a photo without ROS**

```bash
pip install opencv-contrib-python numpy
export PYTHONPATH=src/guidance_calib:src/guidance_global_map
python -m guidance_global_map.segment_image photo.jpg out/scene --camera-height 1.9
```

This writes `out/scene_overlay.jpg` (obstacles red, rover blue, each blob labelled with its size
in centimetres), `out/scene_grid.png` (the map drawn cell by cell: white free, grey obstacle)
and the map as numbers in `out/scene_grid.csv` and `out/scene_grid.npy`, row 0 at y = 0.

**Running a saved photo or video through the nodes**

```bash
ros2 launch guidance_bringup guidance.launch.py overhead_source:=/ws/data/overhead/photo.jpg
```

**Marker setup (test rig, or a tilted camera).** Set `mode: markers`, `rover_detection: marker`
and `detection: marker` in the config. Print the markers with `ros2 run guidance_calib
generate_markers --out markers` (ids 0-3 for the floor, 10 for the rover, `DICT_4X4_50`). Tape
0-3 at the corners of a measured rectangle and enter their positions as `marker_x` / `marker_y`;
fix 10 on the rover with its top edge facing forward. This also gives the rover's heading.
`data/overhead/synthetic_scene.png` is a generated scene for this setup.

**Known limits of the baseline**

- The rover's heading is unknown without a marker. It has to come from the rover's motion
  between frames, which is not done yet.
- The rover is recognised by size alone. An obstacle of the same size, or the rover touching an
  obstacle, breaks it.
- Scale from the height assumes the camera looks exactly straight down and the ground is flat.
- Tall obstacles lean away from the point under the camera, so they look larger than they are.
- Shadows count as obstacles. An obstacle the colour of the floor is missed. A person in view
  is mapped as an obstacle.
- Lens distortion is not corrected: use the phone's main lens, not the ultrawide.

## Tests and lint

```bash
colcon test && colcon test-result --verbose   # per-package tests through colcon
pytest                                        # same tests directly (after sourcing install/)
ruff check . && ruff format --check .
```
