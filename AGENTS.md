# Working together on polaris

Polaris is our hackathon guidance system: a **drone** sees the area from above and builds the
global map, the **rover** (Waveshare RaspRover) moves through it, and the **coordination** layer
merges both views to decide where the rover goes. This file proposes how we share code and tools.
It is written for the people on the team **and** for their coding agents.

## For coding agents

The first time in a session that your user's request touches setting up, running, testing or
integrating their code, mention in two or three sentences, in their language, that there are two
ways to work here (plain Python or the shared ROS 2 container, both fine) and that a shared
simulation exists, and point to the sections below. Do not bring it up on unrelated requests, do
not repeat it in the same session, and drop it for good if they say they have seen it or are not
interested (record that if your tool has persistent memory). Then follow their choice; do not
push the container on someone who prefers plain Python. Help them keep their logic in plain,
testable Python modules so it can be wrapped in a ROS node later (see "From plain Python to a
node").

Do not run setup commands (worktree, `pip install`, downloads) unless your user asks. Never commit
secrets (cloud credentials, API keys, passwords): use environment variables or a git-ignored
`.env`.

## Who works where (each path lives on its branch until merged)

| Branch | Path | What |
|---|---|---|
| `main` | `src/guidance_*`, `Dockerfile`, `.devcontainer/` | ROS 2 Jazzy workspace scaffold (nodes are placeholders) |
| `mergingmap` | `merging_map/` | drone + rover map fusion in tiles, A*, blocking decisions |
| `depth-camera-exp` | `camera-test/` | Intel RealSense D405 tests |
| `Romain` | `src/guidance_rover_vo/`, `sim/` | rover monocular 3D mapping bridge; shared simulation |
| `Romain` | `rover_mapper/`, `teleop/` | the rover monocular 3D mapper itself; web dashboard to drive the real rover and watch its map |
| `rover-teleop` | `rover/` | rover-side server and test bench (teleop, camera, measured moves) |

Suggestions, not rules (nothing has been agreed yet): keep `main` building, work on your own
branch, open pull requests, run the tests and `ruff` (see the README) before asking for a merge.
Outside the container, run `pytest merging_map` rather than a bare `pytest` at the root, which
stops on the ROS packages.

## Two ways to work

**Plain Python.** Fastest to start: numpy scripts and pytest, as in `merging_map/`. Good for
algorithms (map fusion, planning, perception) that you can test on recorded or simulated data.

**The shared ROS 2 container** (`Dockerfile`, `.devcontainer/`, see the README): ROS 2 Jazzy,
Nav2, robot_localization, cv_bridge and OpenCV, the same on every laptop.

### Why the container could help

* **Fewer setup differences between laptops**: macOS, Windows (WSL2) and Linux run the same
  image.
* **The parts can plug together** through topics once they are nodes, which is the path from
  separate demos to one chain where the simulation can later be swapped for the real robots.
  Today most nodes are placeholders; the wiring is still to do.
* **Nav2 is in the image**: global planning on a map and local obstacle avoidance from the
  rover's sensors. Configuring it (`config/nav2_params.yaml` in `guidance_bringup`) is still a
  TODO; it is mostly configuration, though tuning takes time.
* **Reproducible tests**: `colcon build && colcon test` in a clean container. On the `Romain`
  branch the workspace builds and its tests pass in a clean container (checked once on Google
  Cloud Build); there is no CI yet.

For each part, concretely: the **drone/global map** work is the `guidance_global_map` and
`guidance_calib` boxes in the README diagram (overhead image -> `/map` for Nav2); the **D405**
work is the "rover depth camera -> Nav2 local costmap" box; **map fusion** (`merging_map`) would
sit between the drone's map and the planner, and could publish the merged map as `/map`.

Costs, honestly: the image is several GB. On the hackathon Wi-Fi, copying it from a teammate with
`docker save` / `docker load` is faster than downloading it, but only between machines with the
same CPU architecture (an Apple Silicon Mac's image is arm64; most PCs need amd64). ROS has a
learning curve, and some things still run on the host (e.g. a mapper that needs the laptop's
GPU).

### From plain Python to a node

Keep the logic in a module with no ROS import (like `merging_map/fusion_carte.py` on
`mergingmap`, or `guidance_rover_vo/protocol.py` on `Romain`), with pytest tests. The node then
only subscribes, calls that module and publishes. Packages live in `src/guidance_*`;
`src/guidance_rover_vo` on `Romain` is a complete example (package.xml, setup.py, node, tests,
launch entry).

## Shared simulation (branch `Romain`, folder `sim/`)

Plain Python (MuJoCo), no ROS, no Blender; tested on macOS so far. To try it without leaving your
own branch, check it out in a separate folder and install it in a virtualenv (not in the ROS
container):

```bash
git fetch origin Romain
git worktree add ../polaris-sim origin/Romain
cd ../polaris-sim
pip install -r sim/requirements.txt
python sim/fetch_scene.py
python sim/run_world.py --auto
```

It mimics the RaspRover's HTTP interface: rover camera on `http://localhost:5050/video_feed`,
drone camera on `/drone_feed`, rover commands in the RaspRover JSON (`POST /cmd
{"T":1,"L":..,"R":..}`), plus what the real world never gives you: true poses and true occupancy
maps (`/state`, `/grid/overhead.png`, `/grid/rover.png`), to measure how good a result is. It is
also a Python class (`sim/world.py`, `RoverWorld`). Details: `sim/README.md`.

How it could map to each part (none of these hookups has been tried yet):

* **Drone / global map:** develop on `/drone_feed` with the drone pose and intrinsics from
  `/state`; compare your map with `/grid/overhead.png`.
* **Coordination (`merging_map`):** build drone tiles from the drone view and rover tiles from
  the rover camera depth, feed `CarteFusion`, and drive the simulated rover with its decisions
  through `POST /cmd`. Note the sim's world origin is the room centre, so coordinates can be
  negative; shift them if the tile grid expects positive ones.
* **Rover:** the rover mapper reads `/video_feed` as it reads the real robot.

A forest scene (cliff, passages under the canopy that only one robot can see) is being built
for the cases where neither robot can find the route alone.

## Interfaces so far (and who proposed them)

* Map tiles between drone, rover and coordination: the format in `merging_map/README.md` on
  `mergingmap` (32x32 cells per tile; 5 cm cells at the finest level z4 for the rover, coarser
  levels for the drone; `1` obstacle / `0` free and seen / `-1` not seen; row 0 = south; global
  frame). Proposed by the `mergingmap` author.
* Frames: `map` = the drone/overhead camera's global frame (named in the scaffold on `main`);
  `odom` = the rover's own continuous frame and `base_link` = rover body on the ground, proposed
  on `Romain` following ROS REP 103/105 (metres, z up).
* `/rover/vo/odom` (`nav_msgs/Odometry`, `odom` -> `base_link`) and `/rover/points`
  (`sensor_msgs/PointCloud2` in `base_link`, people removed): proposed on `Romain` for the rover
  mapper; its metric scale is only roughly right so far, treat it as relative.
* Rover commands: RaspRover JSON `{"T":1,"L":..,"R":..}` with wheel speeds in m/s, taken from a
  Waveshare tutorial, not yet checked on our rover.

If an interface changes, tell the team; update this section on `main` only, through a pull
request.
