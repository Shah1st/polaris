# State of Polaris

Updated 2026-11-15, evening. Read this first in a new session, human or coding agent: it says
what runs today, how well each part is proven, which settings matter and what is still open.
Update it in the same commit as any change of behaviour. Details of the overhead camera are in
[overhead-camera.md](overhead-camera.md); the step-by-step guide for a real run is in
[demo-reelle.md](demo-reelle.md).

## The chain that runs the demo

```
phone above the area (CamStream app, iphone-camera/)
        │ pictures over Wi-Fi or USB
        ▼
teleop/server.py            the dashboard (web page, relays to the two below)
        │
        ▼
sim/fusion_reelle.py        the maps, the planner (merging_map A*), the missions
        │ HTTP: GET /state, POST /path, /move, /stop
        ▼
rover/rover_server.py       on the rover: motors, odometry, safety
```

`sim/fusion_reelle.py` uses the overhead pipeline (`src/guidance_global_map`) to turn the
phone's picture into the drone map, and `merging_map/` to plan. `scripts/mac/run_local.sh`
starts the whole chain on a Mac. The ROS nodes in `src/` publish the same map on topics, but
the demo chain does not go through ROS.

## How well each part is proven

| Part | Evidence |
|---|---|
| Global navigation from the overhead camera, rover located by its tag, local map off | **Seen working live by the team** (reported on 15 Nov, evening; not observed by the agent who wrote this file) |
| Smaller obstacle growth (15 cm keep-out, 15 cm steps, shadows removed) | **Seen working live by the team**, same report |
| Reading the rover's tag (AprilTag 25h9, id 10) | One real photo (IMG_4109: tag 123 px wide, position, heading and scale read) |
| Obstacle detection from above | Seven real photos of the floor mock-up |
| Vision model that finds the rover (`sim/rover_du_dessus.py`) | 17 real pictures, by its author; not run in the dev container (needs `transformers` and `torch`) |
| Drone map placed by the rover's move, without a tag | Tests only, with rendered pictures |
| Heading from the rover's move in vision-model mode | Tests only, with a stand-in for the model |
| Scale check on the rover's move in `merging_map/lancer_overhead.py` | Tests only, with a simulated rover |
| "Calibrate direction" button | Tests only, with stand-ins (its author's note) |
| Rover's own map (perception service, D405) | Off by default; not checked by this file's author |
| ROS nodes (`/map`, `/rover/pose_overhead`, `merging_map_node`) | Run in the container on one photo; not part of the demo chain |
| iPhone app: 0.5x lens | Builds on GitHub; used by the team |
| iPhone app: LiDAR depth, video stabilisation | Compiled only, never run on a phone (their author's note) |

## Settings that decide what the rover does

Defaults of `sim/fusion_reelle.py` on `main`:

| Setting | Default | Meaning |
|---|---|---|
| `--carte-locale` | off | The rover's own camera builds the local map. Off: planning on the drone map only. |
| `--drone-sans-marqueur` | on | The heading read in the drone's picture (tag, or the white paper seen by the vision model) is ignored; the rover's move places the drone map. Turn it off to place the map from the tag in one picture. |
| `--drone-detecteur` | `marqueur` (the Mac script passes `ia`) | How the rover is found from above: its tag, or the vision model. |
| `--rayon` | 0.15 m | Distance kept between the rover's centre and any obstacle. Two obstacles must be twice this apart. No margin on top. |
| `--pas` | 0.15 m | Length of a mission step. Unused when `--segment` is set (the Mac script sets 2.0). |
| `--drone-ombres` | on | Shadows on the floor are not obstacles. |
| `--drone-hauteur` | 1.8 m | Only a starting point for the scale; the tag or the rover's move corrects it. |
| `--drone-champ` | 108 degrees | Field of view of the phone's lens (the 0.5x lens). Must be right. |
| `--drone-pas-min` | 0.12 m | Move of the rover needed between two stops to place the map without a tag. |

Overhead pipeline (`src/guidance_bringup/config/overhead.yaml`, `OverheadSettings`): cells of
5 cm, a cell touched by an obstacle is an obstacle, camera-side `margin` 0, tag family
`DICT_APRILTAG_25h9`.

The tag: the black square must be about 48 pixels wide in the picture and sharp. Print it with
`python -m guidance_calib.generate_markers --ids 10 --side-cm 12`. Read it with a recent OpenCV
(`pip install opencv-contrib-python`): OpenCV 4.6 in the dev container reads blurred tags badly.

## Dashboard switches (maps page)

| Box | Default | Effect |
|---|---|---|
| Local map (rover camera) | unchecked | Checked: the rover's camera is fused while it stands still and at each stop. Unchecking empties the rover map. |
| No marker on the rover | checked | Unchecked: the tag places the drone map from one picture. Checked: drive the rover at least 12 cm and stop, twice, in view. |

Both take effect on the next picture, without a restart, for everyone on the dashboard.

## Testing

```bash
docker run --rm -v "$PWD":/ws polaris-dev scripts/test_fast.sh   # 200 tests, about 30 s
```

Without Docker: `pip install opencv-contrib-python numpy pytest`, then the same list of test
files with `src/guidance_calib`, `src/guidance_global_map`, `src/guidance_rover_pose`,
`src/guidance_supervisor` and `merging_map` on `PYTHONPATH` (see the script). The full run
(`colcon build`, `colcon test`, `pytest`, then `pytest rover/tests sim`) takes about five
minutes and is for changes to the ROS nodes. `teleop` and `rover_mapper` tests need `requests`
and `torch`, which the image does not have.

One photo, no server: `python -m guidance_global_map.segment_image photo.jpg out/x
--rover-marker 10 --marker-size 0.12 --focal-length-35mm 13`.

## Open decisions

- Whether "No marker on the rover" should now be unchecked by default, since the tag works.
- Which detector is the default: the code says the tag, the Mac script says the vision model.
- Whether the `experiment` branch (phone that reports its own position, `air_ground` module)
  goes into `main`. Not reviewed.
- Turning the local map back on, and which sensor feeds it (rover camera or D405).
- Running the planner on the rover's own computer. Proposed; it would only help when Wi-Fi drops.

## Known limits

- Scenes are small and crowded: with 15 cm of keep-out the rover needs 30 cm between obstacles.
- Without the local map, an obstacle the overhead camera misses is found only by bumping into it.
- Placing the drone map by the rover's move needs a still camera between the two stops.
- A change of lens, resolution or stabilisation mode on the phone changes the scale: let the
  system measure it again (tag or move), never reuse an old number.
