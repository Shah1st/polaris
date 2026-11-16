# Real demo: procedure to follow (15 Nov 2026)

For the team on site this afternoon. **`lancer_overhead.py` (mission with the overhead
camera) has not yet run on the real robot or on a real camera**: the commands were
verified on the VM (`--help`, dry runs on a photo, fake rover), not on the hardware.
What has not been verified is marked *not verified*. What has already driven for real: section 1 bis.
Always keep someone with a hand on Ctrl-C (see "Stopping").

## 1. What runs where

| Program | Where | Why |
|---|---|---|
| Overhead camera (phone with an IP camera app, or USB camera) | in the room, fixed above the arena, lens pointing down | it is the "drone" |
| `merging_map/lancer_overhead.py` (the mission) | **a laptop on the room's Wi-Fi**, the same network as the phone and the rover, or the VM through the robot's tunnels (section 1 bis) | it must read the phone's stream and talk to the rover; the VM sees the phone only through tunnel 18092 |
| `rover/rover_server.py` | on the rover (RDK X5), service `polaris-rover-api` | the only program that talks to the ESP32 |
| `teleop/server.py --suivi` (jury page, `/mission`) | the **same laptop** as the mission | it reads the `--suivi` file written by the mission, locally |
| `rover_mapper/perception_service.py` (optional) | the VM (the GPU is needed) | see step 9; not for the first demo |

## 1 bis. The mission on the VM (what ran on 15 Nov)

The mission can also run **on the VM**. The robot keeps two outgoing SSH tunnels to it:

| On the VM | What |
|---|---|
| `127.0.0.1:18080` | the rover's API (RBDRB's server, see below) |
| `127.0.0.1:18091` | the capture server of the D405 depth camera |
| `127.0.0.1:18092` | reserved for the phone's camera; opened by `/home/romaindcrl/polaris-reel-run/robot_extra_tunnel.sh <IP du téléphone> <port>` |

Scripts in `/home/romaindcrl/polaris-reel-run`: `mission.sh essai` (`--dry-run`) then
`mission.sh vrai` (variables `MPP`, `VICTIME`, `MARQUEUR`, `CAMERA`, `EXTRA`: see the header of the
script).

- **The link is slow**: 1.3 to 4 s per request (`GET /state`), and it dropped twice for a few
  minutes. Pass `--liaison-lente` (HTTP timeout 8 s, link lost only after 15 s without a
  response, no `GET /state` before the expected duration of the step, one read per second). With
  `mission.sh`: `EXTRA="--liaison-lente --delai-http 8"` (the last value wins over the
  `--delai-http 2` of the script, which otherwise takes priority over the one from `--liaison-lente`).
  `--delai-liaison S` sets separately the time without a response before "link lost". With
  `--delai-http 6` alone, `--delai-liaison` stays at 2 s: each lost response becomes
  `LIAISON ROVER PERDUE` (rover link lost) and one more no-go zone.
- **The robot runs RBDRB's newer server**: no `boot_id` or
  `stop_count` in `/state` for now (section 3.4); a STOP given between two steps is
  therefore not seen by the mission.
- **First real autonomous drive** (`polaris-reel-run/avance_esquive.py`): the rover's camera
  alone (perception service on the VM), odometry as position, no overhead camera;
  about 3.5 m in 20 steps going around cardboard boxes, stopped by the operator. It needed
  `delai_liaison=10`, `periode=0.5`, 6 s of client timeout, and giving the perception the real
  pan of the gimbal (-4.5°).
- The complete `lancer_overhead.py` (overhead camera + rover) has not run yet: *not
  verified*.

## 2. Install on the laptop (no ROS)

Python 3.10 or later, `git`, then at the root of the repository (branch `Romain` or the one you are
given):

```bash
pip install numpy opencv-contrib-python requests
```

`opencv-contrib-python` (not `opencv-python`): the ArUco module is needed. `requests` is only used
for the jury page. In each terminal, at the root of the repository, before all the `python`
commands below:

```bash
export PYTHONPATH=merging_map:src/guidance_calib:src/guidance_global_map:src/guidance_rover_pose
```

On Windows (PowerShell): `$env:PYTHONPATH="merging_map;src/guidance_calib;src/guidance_global_map;src/guidance_rover_pose"`
(separator `;`, *not verified*).

## 3. Step by step

### 3.1 Mount the camera and check the stream

Fixed camera, truly vertical, high enough to see the whole arena (the phone's main
lens, not the ultra wide-angle; exposure and focus locked if the app allows it).
Arena: matt floor of one colour, obstacles in bright colours, **nothing else on the floor** (no
tape, cable, bag or ArUco marker on the floor: on the test photo, the markers placed on the
floor come out as obstacles), even light without hard shadows.

```bash
python -m guidance_global_map.measure_scale http://<IP du téléphone>:8080/video --distance 1 --save frame.jpg
```

Source: `http://<IP>:8080/video` (MJPEG stream, an app of the "IP Webcam" kind), or a single-image
URL such as `http://<IP>:8080/shot.jpg`, or `0` for a USB camera on the laptop. Without
`--points`, the command stops and prints `image is LxH; give --points ...`: this is normal,
it has read an image (size printed) and saved it to `frame.jpg`. Check that
`frame.jpg` is recent and shows the whole arena. (Stream reading tested only against local
servers, never against a phone.)

### 3.2 Measure the metres per pixel (with a tape measure)

Lay a tape measure flat in the middle of the view, take two marks 1 m or more apart, note
their pixels in `frame.jpg` (any image viewer), then:

```bash
python -m guidance_global_map.measure_scale frame.jpg --points U1 V1 U2 V2 --distance 1.0
# or, on a laptop with a screen: ... frame.jpg --distance 1.0 --click
```

Verified example: `image 780x630, 333.0 px for 1.0 m` then
`metres_per_pixel: 0.003003  (view 2.34 x 1.89 m)`. The view printed must match
the arena. **Measure again if the stream resolution changes or if the camera moves.** Remove the
tape measure before the mission.

### 3.3 The ArUco marker on the rover

- Id **10**, dictionary `DICT_4X4_50`:
  `python -m guidance_calib.generate_markers --ids 10 --out markers`
  writes `markers/aruco_10.png` (verified).
- Printed at least 10 cm on a side with its white margin, stuck **flat** on the top of the
  rover, **top edge of the marker towards the front of the rover** (no option of
  `lancer_overhead.py` corrects another mounting).
- Always launch the mission with `--marqueur 10`. Without a marker, the rover is looked for as an
  object of its size: on the test photo, a 20 x 26 cm box was taken for the rover
  (0.84, 0.75) while the real rover was at (0.62, 1.33), and the heading is unknown.
- Height of the marker: measure it with a tape measure (floor -> surface of the marker) and give it
  with `--hauteur-marqueur h`, **with `--hauteur H`** (floor -> camera lens, 1.9 m by
  default): the parallax correction uses both. Without `--hauteur-marqueur`, no
  correction: the position seen is shifted outwards by about distance to the centre of
  the image x h / (H - h), that is 8 to 9 cm at 1 m from the centre for h = 0.15 m and H = 1.9 m. Verified
  only on the test photo (the rover goes from (0.62, 1.33) to (0.66, 1.30) with
  `--hauteur-marqueur 0.15`, towards the centre of the image); with a tape measure on the real rover: *not
  verified*.

Check on an image (writes `salle_overlay.jpg`: obstacles in red, rover in blue, nothing
else in red):

```bash
python -m guidance_global_map.segment_image frame.jpg salle --metres-per-pixel 0.0030 --rover-marker 10
```

### 3.4 The rover's server: GET /state

The rover is at `http://<IP du rover>:8080` (IP on the room's Wi-Fi: on the robot,
`hostname -I`). Rover set down, motionless:

```bash
curl -s http://<IP du rover>:8080/state
```

| Field | What it says |
|---|---|
| `link.connected` | the ESP32 answers over USB; `false`: nothing will move |
| `odom.gyro_ready` | gyroscope bias measured; `false` for ~10 s after a start (ESP32 ~7 s + 2 s): **do not touch the rover**; any step is refused (HTTP 409) before that |
| `boot_id`, `uptime_s` | identity of the server process and how long it has been running; a `boot_id` that changes = server restarted. **Absent** = the robot still has the old server (see section 6) |
| `ctrl.mode`, `ctrl.fault` | `idle` at rest; `fault` says why the server cut a step on its own (`no ESP32 feedback`, `ESP32 link lost`, `controller error: ...`); the rover's page shows it in a red banner |
| `stop_count` | number of STOPs received (`POST /stop`, `/cmd {"T":0}`) since the server started; the mission uses it to see a STOP between two steps. **Absent** = old server: that STOP is not seen (section 4) |
| `loop_errors`, `link.handler_errors` | errors of the control loop; must stay at 0 |

(Fields verified on the repository's `rover_server.py --sim`. On 15 Nov, the robot runs
RBDRB's newer server: `boot_id` and `stop_count` are **absent** there; the mission does
without them, and sees a restart from a drop in `odom.odometer` or `link.rx_count`.)

### 3.5 Dry run (`--dry-run`): no POST is sent

```bash
python merging_map/lancer_overhead.py --camera http://<IP du téléphone>:8080/video \
    --metres-par-pixel 0.0030 --marqueur 10 --hauteur <H> --hauteur-marqueur <h> \
    --rover http://<IP du rover>:8080 --victime 1.8 1.5 --suivi $HOME/suivi.json --dry-run
```

`--victime X Y` and `--depart X Y` are in the camera's frame: origin at the **bottom-left
corner of the image**, x to the right, y up, in metres. From a pixel (u, v) of
`frame.jpg` of height H pixels: x = u x mpp, y = (H - v) x mpp. The victim must be inside the
area, otherwise `ValueError ... hors de la zone` (outside the area).

Read the output (example assembled from two verified dry runs on the test photo:
the `Rover joint` line against `rover_server.py --sim`, the others with `--marqueur 10`):

```
Rover joint : liaison ESP32 True, gyroscope pret True, boot_id 122b409f3c60dd72
Zone vue par la camera : 2.34 x 1.89 m (...) ; echelle 3.000 mm/pixel (--metres-par-pixel)
Rover vu par la camera : (0.62, 1.33) m, cap -34 deg
[dry-run] decision : SUIVRE, chemin de 3.88 m
[dry-run] pas 1 : (0.62, 1.33) -> (0.49, 1.19) m, 0.19 m
```

To check: the area is the size of the arena; the rover's position falls within a few cm of the
real one (with a tape measure); the heading is given (otherwise the marker is not seen); the decision
and the first steps make sense; **the rover has not moved**. Without `--dry-run`, the mission starts
by checking the scale on the rover: it moves forward 15 cm then comes back (leave it 15 cm
free in front and behind). With `--metres-par-pixel`, it stops if the rover contradicts it
by more than 10 %; without it, the scale deduced from `--hauteur` is replaced by the one the
rover shows, printed so that it can be reused. `--sans-etalonnage` removes this check
and then requires `--metres-par-pixel`. `--image photo.jpg`
(still photo) is accepted only with `--dry-run` (otherwise refused, code 2).

### 3.6 First real step, hand on the stop

Rover set down in a clear area, victim close (less than 1 m), one person at Ctrl-C,
another near the rover. Same command **without `--dry-run`**, then Ctrl-C as soon as the first step is done.
Check: the rover made a step of about 20 cm in the right direction, it stopped at
Ctrl-C, the output says `Interrompu (Ctrl-C) : /stop envoye au rover.` (interrupted, /stop sent to the rover).

Without a heading (marker not seen), the first movement is a straight line of about 15 cm **in an
unknown direction** (or a pivot then 15 cm backwards if the front is blocked): leave
room all around.

### 3.7 The complete mission

Same command without `--dry-run`. Useful options: `--vitesse 0.15` (m/s, default 0.2),
`--delai-http 2` if the Wi-Fi is slow, `--latence 0.8` if the phone's stream is more than 0.5 s
late (the camera waits for an image that arrived at least `--latence` s after the rover stopped),
`--tolerance 0.15` (arrival within, in m). At the end of the mission, two lines:
`Fin de mission (arrive) : x m parcourus.` (or `(abandon)`, `(erreur)`, `(arret_operateur)`),
then `Issue : <issue> ; raison : <raison>` which says why it stopped (`arrivee`,
`objectif_degrade`, `erreur_robot`, `rover_injoignable`, `sans_progres`, `arret_operateur`...).

### 3.8 The jury page

On the same laptop, in another terminal (`requests` is needed):

```bash
python teleop/server.py --robot http://<IP du rover>:8080 --suivi $HOME/suivi.json --port 8082
```

then open `http://127.0.0.1:8082/mission` in full screen (same absolute path as the `--suivi`
of the mission). For a screen on another machine, `--host 0.0.0.0` opens the server to the room's
network, **including driving the rover (`/api/cmd`)**: prefer plugging the projector
into the laptop.

Rehearse without a robot:

```bash
python teleop/demo_mission.py --port 8150     # then http://127.0.0.1:8150/mission
```

(Verified: `/mission` and `/api/mission` answer 200, and a `--suivi` written by a dry
run of `lancer_overhead.py` is displayed in `/api/mission`. No browser has yet
displayed the page: look at it before the demo.)

### 3.9 Option: the rover's perception (not for the first demo)

`rover_mapper/perception_service.py` needs Depth Anything 3 and the GPU: it runs on the
VM, reads the rover's camera through the VM's dashboard (so through the robot's tunnel), and
listens only on 127.0.0.1. Chain *not verified*:

```bash
# on the VM (robot's tunnel running, teleop dashboard on 8082)
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/perception_service.py \
    --source http://127.0.0.1:8082/video_rect --cam-height 0.22 --fov 100 --port 8095
# on the laptop
ssh -N -L 8095:127.0.0.1:8095 <vous>@<VM>
# then add to the mission: --perception http://127.0.0.1:8095
# and to the jury page: --perception-debug http://127.0.0.1:8095/debug.jpg
```

First measure the real height of the rover's camera (`--cam-height`) and look at
`/debug.jpg` with a box at 0.5 m then 1 m (red = obstacle, green = floor). A response that takes more
than 2 s is ignored: the rover then drives on the overhead map alone, without an error.

### 3.10 Rover alone: move forward while dodging, in one go

`merging_map/avancer_esquive.py` replaces `polaris-reel-run/avance_esquive.py` (section 1 bis):
the rover moves forward `--distance` m straight ahead, going around what **its** camera sees, without
an overhead camera (odometry is used as the position). Made for the slow tunnel: a single
`POST /path` of a few points per segment (1 m at most, only over cells seen free),
no `GET /state` before the expected duration of the segment, one image after each stop, a gimbal
sweep (-40°, +40°, 0) only at the start or when the path is closed or enters
cells not seen. On the VM, robot's tunnel running:

```bash
# the perception service itself reads the robot's raw snapshot and rectifies it (--calib)
/home/romaindcrl/rovermap/.venv/bin/python rover_mapper/perception_service.py \
    --source http://127.0.0.1:18080/snapshot.jpg --calib teleop/calib/plumbline.json \
    --rect-fov 100 --fov 100 --cam-height 0.22 --range 1.5 --port 8095
# first a dry run: perception and first segment, NO POST
python merging_map/avancer_esquive.py --rover http://127.0.0.1:18080 \
    --perception http://127.0.0.1:8095 --distance 3 --essai
# for real, hand on Ctrl-C, /mission page with --suivi
python merging_map/avancer_esquive.py --rover http://127.0.0.1:18080 \
    --perception http://127.0.0.1:8095 --distance 3 --suivi /tmp/suivi.json
```

Variant `--instantane http://127.0.0.1:18080/snapshot.jpg`: the pilot itself downloads
the rover's image (one request per look, `--delai-http`) and sends it to the service by
`POST /obstacles`; the service then does not contact the robot (it keeps `--calib`).
Default settings for the tunnel: `--delai-http 8`, `--delai-liaison 15` (a slow response
is not a cut), `--periode 1`, `--delai-perception 12` (the service waits up to
10 s for the snapshot), `--attente-liaison 180` (cut: it waits for the rover without resending anything),
`--vitesse 0.12`, `--segment 1`, `--demi-largeur 1.5`. End: `arrive`, `bloque`,
`perception_indisponible`, `arret_operateur`, `erreur_rover` or `max_segments`, then the route
and a summary (m driven, segments, requests to the rover, images, s/m). Any end, Ctrl-C (even
twice), SIGTERM and SIGHUP send `/stop`.

Measured on benches (VM, 15 Nov), not on the robot:
- fake server made slow (1.5 s per request, one response in 10 lost with 8 s of waiting,
  1.5 s per image), 5 m with a box on the line: arrived in 6 segments and one gimbal
  sweep, 28 requests to the rover (5.6 per m, 3 or 4 per segment) plus 9 images, that is 7.5
  requests per m in all, 118 s, **24 s/m**. The old step-by-step loop on the same fake server:
  170 requests (35 per m, without the images), 453 s, 92 s/m. The real trial of 15 Nov: about
  37 s per 22 cm step, that is about 168 s/m;
- real `rover_server.py` with a simulated ESP32, 2.2 m at 0.25 m/s, box at 1 m: arrived in
  3 segments, 19 requests to the rover and 6 images, 16 s (7.5 s/m), never closer than 0.20 m
  to the box (radius 0.155 m);
- stops tested: STOP during a segment, keyboard driving between two segments (odometer),
  server restart between two segments (recovered) or during a segment
  (`erreur_rover`, rover stopped), 2 min cut, perception absent (the rover does not move),
  wall across the whole corridor (`bloque`).

*Not verified on the robot*: none of this pilot has driven for real. To watch on the first trial:
the expected duration of a segment (length / speed, pivots at 1 rad/s, + 0.5 s) against the real one;
`gimbal.moving` during the gimbal sweep (on RBDRB's server, it only follows keyboard
driving); the DA3 perception from the rover's real point of view (it masks what is behind
obstacles: more gimbal sweeps than on the bench); the sides of the rover can pass over
cells not seen at the edge of the field of view; a path that starts behind the rover ends in `bloque`. A
server restart right between the `GET /state` and the `POST /path` of a segment is only seen
at the end of the expected duration (about 9 s for 1 m).

During a run:
- **Do not touch the Reset button (odometry) or the gyroscope recalibration on the rover's
  page.** The pilot does not read the state during the expected duration of a segment (about 9 s for
  1 m): a server that does not stop the step before `/reset_odom` (the robot's one on 15 Nov, to be
  checked) keeps pursuing until then the target of the old frame from the new origin, in
  a direction never seen (about 1 m on the bench). The repository's `rover/rover_core.py` stops the
  step first (`reset_odometry`, `recalibrate_gyro`): deploy it (section 6) before the first
  trial of this pilot or of `--liaison-lente`.
- **Stop `sim/fusion_reelle.py` and close the dashboard's stream**: on the same tunnel,
  the fusion reads `GET /state` every 2 s and the perception service takes, on almost every
  cycle, a new 72 kB snapshot, which is more than the pilot's 7.5 or so requests per metre.
- What the pilot does on its own: a segment that ends `stalled`, `timeout` or stopped by the tilt
  guard (`tilt: ...`) marks an obstacle and a no-go zone in front of the rover, then it
  passes beside it; at the 2nd trigger of the guard, `bloque`. After a cut: a STOP, a
  command from another client or keyboard driving during the cut give
  `arret_operateur`, a server restart after the segment was sent `erreur_rover`, and a
  collision during the cut is marked. A lost response to the `POST /path` is resent only if
  a `GET /state` says that the server did not receive it.

## 4. Stopping

- **Ctrl-C on the laptop: the safe stop.** During the mission, it stops it and sends
  `POST /stop` to the rover (3 attempts). (Before the first image, nothing has been sent yet.)
- STOP on the rover's page **during a step**: the mission prints `ARRET OPERATEUR` (operator stop) and
  stops for good (tested on the simulated bench). Holding an arrow on the page (manual takeover)
  does the same.
- STOP on the page **between two steps** (rover motionless): seen **if the rover's server has
  `stop_count` in `/state`** (the server of this repository, not yet on the robot: section 6).
  Before sending each step, the mission compares this counter with the `/stop` it has sent
  itself; if there are more, it prints `ARRET OPERATEUR : STOP recu par le serveur du
  rover entre deux pas ...` and sends nothing more (tested against a fake server and the real
  server with a simulated ESP32, never on the robot). Limits: a STOP pressed between
  launch and the **first** step is not seen; if the response to a `/stop` from the mission is lost
  on the Wi-Fi, the mission may stop as on a STOP (false alarm, on the safe side).
  **Without `stop_count`** (old server on the robot): not seen, the mission will send the next
  step and the rover will set off again; then it has to be Ctrl-C.
- On the robot side: the ESP32 cuts the motors after 3 s without a command, manual driving stops
  0.5 s after the last key, and the server of this repository (not yet on the robot) also
  stops the rover after 0.5 s without feedback from the ESP32.

## 5. When things go wrong

| What you read | Likely cause | What to do |
|---|---|---|
| `Le rover ne repond pas a GET /state` (nothing sent) | wrong IP/port, server stopped, not the same Wi-Fi | `curl http://<IP>:8080/state`; see section 6 |
| `Aucune image de la camera ... en 15 s` | stream URL, phone asleep, another network | open the URL in a browser on the laptop |
| `overhead camera: frame rejected: ...` then `La camera ne donne pas d'image` | image black, overexposed, too dark, almost uniform (`nearly uniform`), hand in front | light, exposure; a very plain empty floor can be rejected: leave an obstacle in view |
| `Rover vu par la camera : NON` | marker not seen (size, glare, blur, not flat), or rover taken for an obstacle | check with `segment_image --rover-marker 10`. `--depart x y` fixes nothing: the first step will fail (`Cap du rover inconnu et camera indisponible`) |
| Area too large or too small, rover shifted | `--metres-par-pixel` wrong, stream resolution changed | redo 3.1 and 3.2 |
| `ERREUR du rover -> arret, nouvel essai : Pas refuse : la cible est a ...` | map <- odometry frame wrong (restart not seen, `/reset_odom`) or camera noise beyond 0.2 m | the mission retries (`/stop`, 1 s) at most 3 times in a row, then stops; if it comes back, Ctrl-C and relaunch. The margin (0.2 m) has no command-line option |
| `Cap du rover impossible a mesurer` | the rover moved neither forwards nor backwards, or the camera does not see it | each attempt redoes the 15 cm straight line; check `gyro_ready`, the wheels, the marker; use `--marqueur 10` |
| `serveur du rover redemarre ...` | `rover_server.py` relaunched (service, another laptop on the serial port) | the mission waits for the ESP32 and the gyroscope, realigns on a fresh image and continues; `... redemarre sans arret : mission arretee` = find who is relaunching the server |
| `Rover pas pret apres 15 s` | ESP32 not connected or gyroscope not calibrated | `GET /state`: `link.connected`, `odom.gyro_ready`; do not move the rover at start-up |
| `LIAISON ROVER PERDUE` then `abandon: rover injoignable` | Wi-Fi cut for more than 2 s; gives up after ~60 s without it coming back | the rover finishes its step then stops on its own; move the router closer, `--delai-http 2` |
| `abandon: le rover n'avance plus` | 8 steps without moving 5 cm, or 4 collisions at the same place | clear the way, check the tracks |
| `ERREUR du rover -> arret, nouvel essai : Pas abandonne par le serveur du rover (statut 'no ESP32 feedback' ...` (or `ESP32 link lost`, `controller error`) | the server stopped the rover on its own; for `no ESP32 feedback` with no visible reason: ESP32 feedback with gaps > 0.5 s on the loaded RDK | the mission retries after a fresh image, then stops in `erreur_robot` after 4 errors in a row; look at `ctrl.fault` and `link.rx_age` in `/state` (red banner on the rover's page); tell Raphaël |
| `ARRET OPERATEUR : STOP recu par le serveur du rover entre deux pas` | someone pressed STOP (rover's page, dashboard), or the response to a `/stop` from the mission was lost | intended: it is over; otherwise relaunch the mission |

The numbers in this table (3 attempts, 0.2 m, 15 cm, 60 s, 8 steps) come from `merging_map/`
(`rover_reel.py`, `cerveau.py`, `mission_settings.py`) as of 15 Nov and may change: Paco is
working on it.

## 6. Deploy the rover's server (with Raphaël)

The robot's server (`/home/sunrise/rover`, started by the `polaris-rover-api` service with
`--fps 40` through a guard script) **differs from the repository** and nobody has compared them. With
Raphaël, and never during a mission:

```bash
scp -r sunrise@<IP du rover>:~/rover /tmp/rover-robot    # keep the robot's version
diff -ru /tmp/rover-robot rover | less                   # carry over what the robot has in addition
```

- Keep the robot's `rover_config.json` (calibrated settings, not in git): do not overwrite it.
- Keep the options specific to the robot (`--fps 40`, serial port...): they are in the service,
  not in the code.
- **Never two programs on the serial port** (opening the port restarts the ESP32;
  a teammate's laptop has already relaunched the server twice). The guard service relaunches a
  server as soon as the port has been free for 5 s: to start `rover_server.py` by hand, first
  `sudo systemctl stop polaris-rover-api`, and `sudo systemctl start polaris-rover-api` afterwards.
- After copying: `sudo systemctl restart polaris-rover-api`, wait for `gyro_ready`, check that
  `/state` contains `boot_id` and `stop_count` (then: STOP on the page, rover motionless ->
  `stop_count` goes up by 1). At start-up, a message `Settings file ... using the default`
  means that a setting of the robot is outside the new bounds: check the settings in the page.
- Then the 6 wheels-off-the-ground tests of `rover/README.md` (section "Deploying to the robot").
- To be checked: that the `SIGTERM` from `systemctl` does reach `rover_server.py` through the guard
  script (otherwise the rover is not stopped when the service stops).

## 7. Written, tested on the VM, but never on the real robot or a real camera

- The whole chain `lancer_overhead.py` -> `rover_reel.py` -> `rover_server.py` for real: only
  against a fake server and the real server with a simulated ESP32.
- Reading a phone's stream (`camera_reader`) and the `--latence` setting; the scale with the
  tape measure; detection of the ArUco marker on the real rover; rejection of bad images
  under the room's lighting.
- The operator stop (STOP on the page during a step or between two steps with `stop_count`,
  manual takeover), detection of a server restart, the bounded step (`Pas refuse`), the
  measurement of the heading by a straight line, recovery after a server failure (`no ESP32
  feedback`...), the `--hauteur-marqueur` parallax correction.
- The rover server of this repository: monitoring of the ESP32 (0.5 s), refusal before `gyro_ready`,
  validation of the numbers and of `/config`, `boot_id`, `stop_count`, stop on `SIGTERM`,
  `/snapshot.jpg` with `X-Age-Ms`, the red fault banner of its page (served, never
  displayed in a browser). Not yet deployed on the robot.
- The `/mission` page in a browser, and on a real mission.
- The perception service on the real rover's camera (evaluated only in simulation) and its
  `--perception` hookup.
- The rover getting out when near an obstacle, the detection of no progress against real noise, the
  thresholds for the rover's radius (0.155 m) and for the arrival tolerance.
- The `devices/` layer (fleet of devices): never tried on the robot; the path tested
  for the demo is `lancer_overhead.py`.

## 8. How this guide was verified (VM, 15 Nov)

`--help` of `lancer_overhead.py`, `measure_scale`, `segment_image`, `generate_markers`,
`rover_server.py`, `teleop/server.py`, `demo_mission.py`, `perception_service.py`;
`lancer_overhead.py --dry-run` on `data/overhead/synthetic_scene.png` (with and without
`--marqueur 10`, without a rover and with `rover_server.py --sim`); refusal of `--image` without
`--dry-run`; a black image (rejection printed); `measure_scale --points` and `--save` on the same
photo; `GET /state` of `rover_server.py --sim`; `demo_mission.py` and `teleop/server.py --suivi`
(`/mission`, `/api/mission`).

Reread after the merge of `fin/integration` (safety, end-to-end bench, leftovers): options and
messages quoted verified in the merged code, `--hauteur-marqueur` tried in `--dry-run` on the
test photo.
