"""Record one real D405 picture with the grid the rover makes of it, for the checks of
docs/local-navigation.md ("Before the first real run", step 4).

    python rover_mapper/scripts/record_d405.py NAME [--d405 http://127.0.0.1:8091] [--out DIR]

Reads the rover's capture server (rover_mapper/d405_server.py, through an SSH tunnel:
`ssh -N -L 8091:127.0.0.1:8091 <user>@<rover>`), never the rover's motors. Writes in DIR
(default rover_mapper/recordings/):

    NAME.npz        GET /frame?fresh=1 as received: colour JPEG, raw 16-bit depth, intrinsics
    NAME.grid.json  GET /grid as received (what the rover's guard would use), the floor
                    calibration and the server's health (cpu_ms_per_grid)
    NAME.jpg        the colour picture and the depth, each with the grid drawn over it (red:
                    obstacle cells, green: free cells), and the numbers

The grid drawn is computed here from NAME.npz with the server's own rule (GridMaker.compute:
same module, same settings, the calibrated floor when there is one), so the picture and its
grid are the same instant; the grid received from the rover is compared with it.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from rovermap import perception as per  # noqa: E402

# GridMaker's defaults (rover_mapper/d405_server.py): keep them the same.
CAM_HEIGHT, STRIDE, MAX_RANGE, BAND = 0.15, 2, 1.25, (0.04, 0.40)
DEPTH_MIN, DEPTH_MAX = 0.07, 3.0


def get(url: str, timeout: float = 6.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def get_json(url: str):
    try:
        return json.loads(get(url))
    except Exception as e:  # noqa: BLE001 - recorded as it is
        return {'error': f'{type(e).__name__}: {e}'[:200]}


def depth_picture(depth: np.ndarray, far: float) -> np.ndarray:
    """Depth in metres as colours: red near, blue at `far` and beyond, black = no measure."""
    near = np.clip((depth - DEPTH_MIN) / (far - DEPTH_MIN), 0.0, 1.0)
    img = cv2.applyColorMap((255 * (1.0 - near)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth <= 0] = 0
    return img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('name', help='name of the scene, e.g. box_030 (no extension)')
    ap.add_argument('--d405', default='http://127.0.0.1:8091')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(HERE), 'recordings'))
    a = ap.parse_args(argv)
    base = a.d405.rstrip('/')
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, a.name)

    raw = get(base + '/frame?fresh=1')
    received = get_json(base + '/grid')
    calibration = (get_json(base + '/calibration') or {}).get('calibration')
    health = get_json(base + '/health')
    with open(stem + '.npz', 'wb') as f:
        f.write(raw)

    z = np.load(io.BytesIO(raw))
    color = cv2.imdecode(z['color_jpg'], cv2.IMREAD_COLOR)
    units = cv2.imdecode(z['depth_png'], cv2.IMREAD_UNCHANGED)
    depth = units.astype(np.float32) * np.float32(float(z['depth_scale']))
    depth[(depth < DEPTH_MIN) | (depth > DEPTH_MAX)] = 0
    K = np.asarray(z['K_depth'], np.float64)
    K_grid = K.copy()
    K_grid[:2] /= STRIDE
    plane = None
    if calibration:
        plane = (np.asarray(calibration['normal'], np.float64), float(calibration['d']))
    p = per.perceive(depth[::STRIDE, ::STRIDE], K_grid, cam_height=CAM_HEIGHT, plane=plane,
                     scale=1.0, max_range=MAX_RANGE, band=BAND)  # fmt: skip
    cells = np.asarray(p.cells)
    here = {'obstacle': int((cells == 1).sum()), 'free': int((cells == 0).sum())}
    theirs = None
    if isinstance(received.get('cells'), list):
        c = np.asarray(received['cells'])
        theirs = {'obstacle': int((c == 1).sum()), 'free': int((c == 0).sum())}
    with open(stem + '.grid.json', 'w', encoding='utf-8') as f:
        json.dump({'grid': received, 'calibration': calibration, 'health': health,
                   'recomputed': {**here, 'floor': bool(p.floor), 'stats': p.stats}}, f)  # fmt: skip

    # The depth is in the depth camera's frame; on the D405 the colour picture comes from the
    # same imager (depth -> colour is the identity within 0.1 mm), so the grid is drawn on the
    # colour picture with the colour intrinsics.
    Kc = np.asarray(z['K_color'], np.float64)
    left = per.draw(color, p, Kc, MAX_RANGE)
    right = per.draw(depth_picture(depth, MAX_RANGE), p, K, MAX_RANGE)
    if right.shape[:2] != left.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    lines = [
        f'{a.name}: floor {"calibrated" if plane is not None else "fitted" if p.floor else "NOT FOUND"}'
        + (f', lens {p.stats.get("floor_height_m")} m up, tilt {p.stats.get("floor_tilt_deg")} deg'
           if p.floor else ''),
        f'grid here: {here["obstacle"]} obstacle, {here["free"]} free cells'
        + (f' | from the rover: {theirs["obstacle"]} obstacle, {theirs["free"]} free'
           if theirs else f' | from the rover: {received.get("error", "no grid")}'),
    ]  # fmt: skip
    band = np.zeros((26 * len(lines) + 8, left.shape[1] * 2, 3), np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(band, text, (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)  # fmt: skip
    # not cv2.imwrite: on Windows it cannot write to a path with non-ASCII characters
    ok, jpg = cv2.imencode('.jpg', np.vstack([np.hstack([left, right]), band]))
    with open(stem + '.jpg', 'wb') as f:
        f.write(jpg.tobytes() if ok else b'')
    print('\n'.join(lines))
    print('written:', stem + '.npz', stem + '.grid.json', stem + '.jpg')
    return 0


if __name__ == '__main__':
    sys.exit(main())
