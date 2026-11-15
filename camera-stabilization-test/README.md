# Camera stabilization test

Small standalone application that reads a webcam or video file, processes frames with
`python_video_stab`, and serves raw and stabilized MJPEG streams side by side.

## Setup

```bash
cd camera-stabilization-test
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Automatically select the first webcam that returns a frame:

```bash
python app.py
```

Or select a specific camera:

```bash
python app.py --source 1
```

The default mode targets a moving drone: a 24-frame smoothing window damps stronger motion,
and the frame is zoomed by 1.35 before stabilization so takeoff and movement have more room
without exposing large black borders. Tune `--smoothing-window` and `--zoom`.

Video file:

```bash
python app.py --source path/to/video.mp4
```

Open <http://127.0.0.1:8085>.

The application has no ROS dependency. `stabilization/` contains the reusable source,
stabilizer adapter, frame models, and processing pipeline; `web/` is only the presentation
layer.

## Test

```bash
pytest
```
