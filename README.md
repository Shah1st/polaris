# Polaris: drone/rover guidance POC

Hackathon proof of concept for a guidance system. An overhead 2D camera (on a drone or a fixed
mount) sees the whole area and builds the global map, and the rover's position comes from the
same camera. A rover with a 3D depth camera follows the planned path and avoids local obstacles
on its own. Everything runs on one laptop with ROS 2 Jazzy.

**Status:** workspace scaffold only. Every node starts and logs, and does nothing else. `TODO(...)`
comments mark where each piece of logic will go.

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
cv_bridge, OpenCV, pytest and ruff.

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

## Tests and lint

```bash
colcon test && colcon test-result --verbose   # per-package tests through colcon
pytest                                        # same tests directly (after sourcing install/)
ruff check . && ruff format --check .
```
