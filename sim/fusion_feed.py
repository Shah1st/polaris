"""Feeding side of the live map fusion on the forest simulation. Plain Python, no ROS, no I/O.

`sim/fusion_live.py` runs Paco's `merging_map` (`CarteFusion`) on live simulator data. This
module holds the parts that do not talk to the network, so they can be tested:

* the drone's grid: the simulator's `/grid/overhead.png` (what the drone sees from its survey
  altitude; under the canopy it sees nothing) -> a -1/0/1 matrix in the fusion's convention;
* the rover's perception: camera depth + intrinsics `K` + camera-to-world pose `T` -> world
  points within a sensor range -> a -1/0/1 matrix on the global 5 cm grid, which `decouper()`
  turns into rover tiles;
* the three pictures shown on the dashboard's /sim page.

Fusion convention (merging_map/README.md): `1` obstacle, `0` free and seen, `-1` not seen;
`matrix[i][j]` is the cell at X = x0 + j * res, Y = y0 + i * res, so row 0 is south.
"""

import io
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

UNSEEN, FREE, OBSTACLE = -1, 0, 1
POINT_GROUND, POINT_OBSTACLE, POINT_HIGH = 0, 1, 2
RES = 0.05  # finest cell of the fusion (level z = 4), also the simulator's grid cell
# Rover camera above the ground under the wheels, forest sim (forest_world.py: BASE_Z plus the
# camera's z in the body, 0.03325 + 0.178382 m).
CAM_HEIGHT = 0.2116


# ----------------------------------------------------------------------------------------------
# Drone side
# ----------------------------------------------------------------------------------------------
def drone_grid_from_png(png: bytes) -> np.ndarray:
    """The simulator's `/grid/overhead.png` as a fusion matrix (row 0 = south).

    The picture is the drone's label map (forest_world.grid('overhead')) drawn x right, y up:
    0 = free, 128 = not seen (under the canopy), 255 = steep, obstacle or water.
    """
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('not a PNG picture')
    out = np.full(img.shape, UNSEEN, np.int8)
    out[img < 64] = FREE
    out[img > 192] = OBSTACLE
    return np.ascontiguousarray(out[::-1])


# ----------------------------------------------------------------------------------------------
# Rover side
# ----------------------------------------------------------------------------------------------
@dataclass
class Scan:
    """One rover observation, whatever produced it (simulator depth today, mapper bridge later).

    `points`: (N, 3) world points the sensor returned within its range. `drops`: (M, 2) world
    x, y where the ground should have been seen but was not, because it falls away (cliff or
    drop edge); empty when the source has no rays (a plain point cloud).
    """

    t: float
    pose: tuple[float, float, float]  # rover x, y, yaw (world)
    points: np.ndarray
    ground_z: float  # height of the ground under the rover
    drops: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    sensor_range: float = 2.0


def scan_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    pose: tuple[float, float, float],
    t: float,
    max_range: float = 2.0,
    cam_height: float = CAM_HEIGHT,
    drop: float = 0.3,
    stride: int = 1,
) -> Scan:
    """Back-project a z-depth image (metres along the optical axis) into world points.

    `T` is camera->world with OpenCV axes (x right, y down, z forward), as the simulator gives
    it. Points farther than `max_range` from the camera are dropped (a real rover sensor is only
    trusted that far). A pixel whose ray should meet the rover's own ground plane within range,
    but whose depth lands more than `drop` m below that plane, marks a drop at the place where
    the ground was expected.
    """
    d = np.asarray(depth, np.float32)[::stride, ::stride]
    h, w = d.shape
    vs, us = np.mgrid[0:h, 0:w].astype(np.float32) * stride
    R, c = T[:3, :3].astype(np.float32), T[:3, 3].astype(np.float32)
    rays = np.stack(
        [(us - K[0, 2]) / K[0, 0], (vs - K[1, 2]) / K[1, 1], np.ones_like(d)], -1
    ).reshape(-1, 3)
    z = d.reshape(-1)
    norm = np.sqrt((rays * rays).sum(1))
    valid = np.isfinite(z) & (z > 0.02)
    near = valid & (norm * z <= max_range)
    # world direction of each pixel, scaled so the optical-axis component is 1
    points = c + (rays[near] @ R.T) * z[near, None]

    up = -R[:, 1]  # the camera's -y axis is the rover's up (camera mounted level)
    g0 = c - up * cam_height
    # depth at which each ray meets the rover's ground plane (camera height below the camera)
    den = rays @ (R.T @ up)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(den < -1e-6, cam_height / -den, np.inf).astype(np.float32)
    # the ray lands farther than the plane, and that far point is well below it
    cand = valid & (norm * s <= max_range) & (z > s)
    dirs = rays[cand] @ R.T
    below = (c + dirs * z[cand, None] - g0) @ up < -drop
    G = c + dirs[below] * s[cand][below, None]
    return Scan(
        t=float(t),
        pose=tuple(float(v) for v in pose),
        points=points,
        ground_z=float(g0[2]),
        drops=G[:, :2],
        sensor_range=float(max_range),
    )


@dataclass
class LocalGrid:
    """A -1/0/1 matrix on the global grid; cell [0][0] has its south-west corner at (x0, y0)."""

    data: np.ndarray
    x0: float
    y0: float
    stats: dict
    # per point of the scan: POINT_GROUND, POINT_OBSTACLE (in the band), POINT_HIGH (above it)
    point_kind: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))


def classify_scan(
    scan: Scan,
    res: float = RES,
    band: tuple[float, float] = (0.05, 0.30),
    ground_radius: int = 2,
    footprint: float = 0.12,
) -> LocalGrid:
    """What one scan says about each 5 cm cell around the rover.

    * Ground of a cell = lowest point in the cells within `ground_radius` cells (so the top of a
      log or a trunk face is measured against the ground next to it).
    * Obstacle (1) = a cell holding a point between `band[0]` and `band[1]` m above that ground:
      something the rover would hit (trunk, rock, log, cliff face, slope over about 25 degrees).
      Points higher up (branches, canopy) do not block: the rover passes under them.
    * Drop (1) = a cell with no point where the ground should have been (see `scan_from_depth`).
    * Free (0) = any other cell holding points; the cells under the rover itself are free too.
    * Not seen (-1) = everything else, including cells hidden behind obstacles.
    """
    x, y, _ = scan.pose
    r = scan.sensor_range + 2 * res
    gi0, gj0 = math.floor((y - r) / res), math.floor((x - r) / res)
    n_rows, n_cols = math.floor((y + r) / res) - gi0 + 1, math.floor((x + r) / res) - gj0 + 1
    out = np.full((n_rows, n_cols), UNSEEN, np.int8)

    p = scan.points
    i = np.floor(p[:, 1] / res).astype(np.int64) - gi0
    j = np.floor(p[:, 0] / res).astype(np.int64) - gj0
    inside = (i >= 0) & (i < n_rows) & (j >= 0) & (j < n_cols)
    i, j, pz = i[inside], j[inside], p[inside, 2]

    zmin = np.full((n_rows, n_cols), np.inf, np.float32)
    np.minimum.at(zmin, (i, j), pz.astype(np.float32))
    seen = np.isfinite(zmin)
    k = 2 * ground_radius + 1
    ground = cv2.erode(
        np.where(seen, zmin, np.float32(1e6)), np.ones((k, k), np.uint8), borderValue=1e6
    )
    rel = pz - ground[i, j]
    hit = (rel >= band[0]) & (rel <= band[1])
    kind = np.full(len(p), POINT_GROUND, np.uint8)
    kind[np.nonzero(inside)[0][hit]] = POINT_OBSTACLE
    kind[np.nonzero(inside)[0][rel > band[1]]] = POINT_HIGH
    obstacle = np.zeros((n_rows, n_cols), bool)
    obstacle[i[hit], j[hit]] = True

    out[seen] = FREE
    # the rover stands on its own cells
    rows = (np.arange(n_rows) + gi0 + 0.5) * res
    cols = (np.arange(n_cols) + gj0 + 0.5) * res
    under = (cols[None, :] - x) ** 2 + (rows[:, None] - y) ** 2 <= footprint**2
    out[under & ~seen] = FREE

    n_drop = 0
    if len(scan.drops):
        di = np.floor(scan.drops[:, 1] / res).astype(np.int64) - gi0
        dj = np.floor(scan.drops[:, 0] / res).astype(np.int64) - gj0
        ok = (di >= 0) & (di < n_rows) & (dj >= 0) & (dj < n_cols)
        drop = np.zeros((n_rows, n_cols), bool)
        drop[di[ok], dj[ok]] = True
        drop &= ~seen & ~under
        out[drop] = OBSTACLE
        n_drop = int(drop.sum())
    out[obstacle] = OBSTACLE
    stats = {
        'points': int(len(p)),
        'free': int((out == FREE).sum()),
        'obstacle': int(obstacle.sum()),
        'drop': n_drop,
    }
    return LocalGrid(out, gj0 * res, gi0 * res, stats, kind)


# ----------------------------------------------------------------------------------------------
# Pictures (BGR, north up, x to the right in all three)
# ----------------------------------------------------------------------------------------------
def _bgr(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[k : k + 2], 16) for k in (5, 3, 1))


# Same colours as merging_map/demo_fusion.py (C_INCONNU, C_LIBRE, ...), so the pictures read like
# the fusion's own demo.
UNKNOWN_C = _bgr('#e1e0d9')
FREE_C = _bgr('#fcfcfb')
ROVER_FREE_C = _bgr('#d3eee2')
DRONE_OBST_C = _bgr('#52514e')
ROVER_OBST_C = _bgr('#eb6834')
PATH_C = _bgr('#2a78d6')
ZOOM_C = _bgr('#6250d6')
INK_C = _bgr('#0b0b0b')
GOAL_C = _bgr('#e34948')
DROP_C = _bgr('#b0177a')
WINDOW_C = _bgr('#1f9bd1')
SCAN_FREE_C = _bgr('#8fd4b4')

# Class codes in priority order: a coarse pixel shows the most important class it covers.
K_UNKNOWN, K_FREE, K_ROVER_FREE, K_DRONE_OBST, K_ROVER_OBST, K_DROP = range(6)
PALETTE = np.array([UNKNOWN_C, FREE_C, ROVER_FREE_C, DRONE_OBST_C, ROVER_OBST_C, DROP_C], np.uint8)


def reduce_classes(classes: np.ndarray, k: int) -> np.ndarray:
    """k x k blocks -> one cell holding the highest class code of the block."""
    if k == 1:
        return classes
    nh, nw = classes.shape
    pad = np.zeros((-(-nh // k) * k, -(-nw // k) * k), classes.dtype)
    pad[:nh, :nw] = classes
    out = pad[::k, ::k].copy()
    for di in range(k):
        for dj in range(k):
            np.maximum(out, pad[di::k, dj::k], out=out)
    return out


@dataclass
class View:
    """A north-up picture of the world rectangle (x0, y0) - (x1, y1) at `ppm` pixels per metre."""

    x0: float
    y0: float
    x1: float
    y1: float
    ppm: float

    @property
    def size(self) -> tuple[int, int]:
        return round((self.x1 - self.x0) * self.ppm), round((self.y1 - self.y0) * self.ppm)

    def px(self, x: float, y: float) -> tuple[int, int]:
        return round((x - self.x0) * self.ppm), round((self.y1 - y) * self.ppm)

    def raster(self, classes: np.ndarray, gx0: float, gy0: float, res: float = RES) -> np.ndarray:
        """Draw a class matrix (row 0 = south, cell [0][0] corner at gx0, gy0) into this view."""
        w, h = self.size
        cols = np.floor((self.x0 + (np.arange(w) + 0.5) / self.ppm - gx0) / res).astype(int)
        rows = np.floor((self.y1 - (np.arange(h) + 0.5) / self.ppm - gy0) / res).astype(int)
        ok_c = (cols >= 0) & (cols < classes.shape[1])
        ok_r = (rows >= 0) & (rows < classes.shape[0])
        out = np.full((h, w), K_UNKNOWN, classes.dtype)
        sub = classes[np.ix_(rows[ok_r], cols[ok_c])]
        out[np.ix_(np.nonzero(ok_r)[0], np.nonzero(ok_c)[0])] = sub
        return PALETTE[out]


def fused_classes(val: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Class codes of the fusion's dense grid (`Grille.val`, `Grille.src`: 1 drone, 2 rover)."""
    out = np.full(val.shape, K_UNKNOWN, np.uint8)
    out[val == FREE] = K_FREE
    out[(val == FREE) & (src == 2)] = K_ROVER_FREE
    out[(val == OBSTACLE) & (src == 1)] = K_DRONE_OBST
    out[(val == OBSTACLE) & (src == 2)] = K_ROVER_OBST
    return out


def drone_classes(m: np.ndarray) -> np.ndarray:
    out = np.full(m.shape, K_UNKNOWN, np.uint8)
    out[m == FREE] = K_FREE
    out[m == OBSTACLE] = K_DRONE_OBST
    return out


def draw_rover(img: np.ndarray, view: View, pose, size_px: int = 9) -> None:
    x, y, yaw = pose
    cx, cy = view.px(x, y)
    pts = []  # an arrowhead with a notch at the back, so the heading cannot be misread
    for a, r in ((0.0, 1.0), (2.45, 0.8), (math.pi, 0.25), (-2.45, 0.8)):
        pts.append(
            (
                round(cx + size_px * r * math.cos(yaw + a)),
                round(cy - size_px * r * math.sin(yaw + a)),
            )
        )
    pts = np.array(pts, np.int32)
    cv2.fillPoly(img, [pts], INK_C, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)


def draw_star(img: np.ndarray, view: View, x: float, y: float, r: int = 9) -> None:
    cx, cy = view.px(x, y)
    pts = [
        (
            round(cx + (r if k % 2 == 0 else r * 0.45) * math.sin(k * math.pi / 5)),
            round(cy - (r if k % 2 == 0 else r * 0.45) * math.cos(k * math.pi / 5)),
        )
        for k in range(10)
    ]
    cv2.fillPoly(img, [np.array(pts, np.int32)], GOAL_C, cv2.LINE_AA)
    cv2.polylines(img, [np.array(pts, np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)


def draw_polyline(img, view: View, pts, colour, thickness: int = 2) -> None:
    if pts is not None and len(pts) >= 2:
        arr = np.array([view.px(x, y) for x, y in pts], np.int32)
        cv2.polylines(img, [arr], False, colour, thickness, cv2.LINE_AA)


def draw_rect(img, view: View, x0, y0, x1, y1, colour, thickness: int = 1) -> None:
    cv2.rectangle(img, view.px(x0, y1), view.px(x1, y0), colour, thickness, cv2.LINE_AA)


def draw_frame(img: np.ndarray, view: View, step: float, label_every: int = 0) -> None:
    """Faint grid lines every `step` m, a north arrow and a scale bar."""
    h, w = img.shape[:2]
    for gx in np.arange(math.ceil(view.x0 / step) * step, view.x1, step):
        px = view.px(gx, 0)[0]
        img[:, px : px + 1] = (img[:, px : px + 1] * 0.85).astype(np.uint8)
    for gy in np.arange(math.ceil(view.y0 / step) * step, view.y1, step):
        py = view.px(0, gy)[1]
        img[py : py + 1, :] = (img[py : py + 1, :] * 0.85).astype(np.uint8)
    # north arrow, top right
    ax, ay = w - 16, 30
    cv2.arrowedLine(img, (ax, ay), (ax, ay - 20), INK_C, 2, cv2.LINE_AA, tipLength=0.4)
    cv2.putText(img, 'N', (ax - 5, ay + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK_C, 1, cv2.LINE_AA)
    # scale bar, bottom left: `step` metres
    bar = round(step * view.ppm)
    cv2.rectangle(img, (8, h - 14), (8 + bar, h - 9), INK_C, -1)
    text = f'{step:g} m'
    cv2.putText(
        img, text, (12 + bar, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK_C, 1, cv2.LINE_AA
    )


POINT_C = np.array([(120, 140, 110), (40, 40, 220), (200, 120, 40)], np.uint8)  # BGR


def render_rover_view(
    scan: Scan,
    local: LocalGrid,
    rover_layer: tuple[np.ndarray, float, float] | None,
    half: float = 4.0,
    ppm: float = 60.0,
    max_points: int = 2500,
) -> tuple[np.ndarray, View]:
    """Panel 1: the rover's last scan seen from above, `half` m around it.

    Faint: every cell the rover has reported so far (its layer in the fusion). Strong: the cells
    of this scan (free, obstacle, drop). Dots: some of the scan's points, by their height above
    the local ground: ground (grey-green), 5-30 cm (red, makes an obstacle), higher (blue,
    branches the rover passes under).
    """
    x, y, yaw = scan.pose
    view = View(x - half, y - half, x + half, y + half, ppm)
    w, h = view.size
    img = np.full((h, w, 3), UNKNOWN_C, np.uint8)
    if rover_layer is not None:
        val, gx0, gy0 = rover_layer
        cls = np.full(val.shape, K_UNKNOWN, np.uint8)
        cls[val == FREE] = K_ROVER_FREE
        cls[val == OBSTACLE] = K_ROVER_OBST
        layer = view.raster(cls, gx0, gy0)
        mask = (layer != np.array(UNKNOWN_C, np.uint8)).any(-1)
        img[mask] = (0.45 * layer[mask] + 0.55 * np.array(UNKNOWN_C)).astype(np.uint8)
    cls = np.full(local.data.shape, K_UNKNOWN, np.uint8)
    cls[local.data == FREE] = K_ROVER_FREE
    cls[local.data == OBSTACLE] = K_DROP
    pts = scan.points
    i = np.floor(pts[:, 1] / RES).astype(np.int64) - round(local.y0 / RES)
    j = np.floor(pts[:, 0] / RES).astype(np.int64) - round(local.x0 / RES)
    ok = (i >= 0) & (i < cls.shape[0]) & (j >= 0) & (j < cls.shape[1])
    # obstacle cells holding points are real obstacles; obstacle cells without points are drops
    has_pts = np.zeros(cls.shape, bool)
    has_pts[i[ok], j[ok]] = True
    cls[(local.data == OBSTACLE) & has_pts] = K_ROVER_OBST
    layer = view.raster(cls, local.x0, local.y0)
    mask = (layer != np.array(UNKNOWN_C, np.uint8)).any(-1)
    free_now = (layer == np.array(ROVER_FREE_C, np.uint8)).all(-1)
    img[mask] = layer[mask]
    img[free_now] = SCAN_FREE_C
    draw_frame(img, view, 1.0)

    # sensor range and field of view
    cx, cy = view.px(x, y)
    rr = round(scan.sensor_range * ppm)
    cv2.circle(img, (cx, cy), rr, WINDOW_C, 1, cv2.LINE_AA)
    if len(pts):
        sel, kind = pts, local.point_kind
        if len(sel) > max_points:
            pick = np.random.default_rng(0).choice(len(sel), max_points, replace=False)
            sel, kind = sel[pick], kind[pick]
        u = np.round((sel[:, 0] - view.x0) * ppm).astype(int)
        v = np.round((view.y1 - sel[:, 1]) * ppm).astype(int)
        inside = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        img[v[inside], u[inside]] = POINT_C[kind[inside]]
    draw_rover(img, view, scan.pose, size_px=12)
    return img, view


def render_drone_view(
    drone: np.ndarray,
    origin: tuple[float, float],
    view: View,
    pose=None,
    goal=None,
    start=None,
    window=None,
) -> np.ndarray:
    """Panel 2: the drone's grid, as fed to the fusion."""
    cls = reduce_classes(drone_classes(drone), max(1, int(1 / (view.ppm * RES))))
    k = max(1, int(1 / (view.ppm * RES)))
    img = view.raster(cls, origin[0], origin[1], RES * k)
    draw_frame(img, view, 10.0)
    draw_markers(img, view, pose, goal, start, window)
    return img


def render_fused_view(
    val: np.ndarray,
    src: np.ndarray,
    gx0: float,
    gy0: float,
    view: View,
    pose=None,
    goal=None,
    start=None,
    window=None,
    path=None,
    trail=None,
    tiles=(),
    tile_rect=None,
    point=None,
) -> np.ndarray:
    """Panel 3: the fusion's merged grid with the planned path and the current decision."""
    k = max(1, int(1 / (view.ppm * RES)))
    cls = reduce_classes(fused_classes(val, src), k)
    img = view.raster(cls, gx0, gy0, RES * k)
    draw_frame(img, view, 10.0)
    for key in tiles:
        x0, y0, side = tile_rect(key)
        draw_rect(img, view, x0, y0, x0 + side, y0 + side, ZOOM_C, 2)
    draw_polyline(img, view, trail, (90, 90, 90), 1)
    draw_polyline(img, view, path, PATH_C, 2)
    if point is not None:
        cx, cy = view.px(*point)
        cv2.drawMarker(img, (cx, cy), PATH_C, cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
    draw_markers(img, view, pose, goal, start, window)
    return img


def draw_markers(img, view, pose, goal, start, window) -> None:
    if window is not None:
        draw_rect(img, view, *window, WINDOW_C, 1)
    if start is not None:
        cv2.circle(img, view.px(*start), 5, INK_C, 1, cv2.LINE_AA)
    if goal is not None:
        draw_star(img, view, *goal)
    if pose is not None:
        draw_rover(img, view, pose)


# ----------------------------------------------------------------------------------------------
# Depth comparison
# ----------------------------------------------------------------------------------------------
NO_MEASURE_C = (0, 0, 0)  # the sensor gave nothing
BEYOND_C = (70, 70, 70)  # a value farther than the colour scale


def depth_stats(sensor: np.ndarray, truth: np.ndarray, d_range=(0.07, 1.0)) -> dict:
    """How a depth sensor compares with the exact depth of the same pixels.

    `sensor`: 0 (or not finite) = no measurement. Returns the share of pixels measured, the
    share measured among pixels whose true depth is within the sensor's range, and the median
    absolute (m) and relative errors on pixels both have.
    """
    s = np.asarray(sensor, np.float64)
    t = np.asarray(truth, np.float64)
    measured = np.isfinite(s) & (s > 0)
    in_range = np.isfinite(t) & (t >= d_range[0]) & (t <= d_range[1])
    both = measured & np.isfinite(t) & (t > 0)
    err = np.abs(s[both] - t[both])
    return {
        'mesures_pct': round(100.0 * measured.mean(), 1),
        'mesures_dans_portee_pct': (
            round(100.0 * (measured & in_range).sum() / in_range.sum(), 1)
            if in_range.any()
            else None
        ),
        'erreur_mediane_m': round(float(np.median(err)), 4) if err.size else None,
        'erreur_mediane_rel_pct': (
            round(100.0 * float(np.median(err / t[both])), 2) if err.size else None
        ),
        'pixels_communs': int(both.sum()),
    }


def colour_depth(depth: np.ndarray, d_min: float, d_max: float, width: int) -> np.ndarray:
    """False colours (near = red, far = blue) on a fixed scale; no measurement = black, farther
    than `d_max` = dark grey."""
    d = np.asarray(depth, np.float32)
    h = round(d.shape[0] * width / d.shape[1])
    d = cv2.resize(d, (width, h), interpolation=cv2.INTER_NEAREST)
    measured = np.isfinite(d) & (d > 0)
    ok = measured & (d <= d_max)
    v = np.clip((d - d_min) / (d_max - d_min), 0.0, 1.0)
    img = np.zeros((h, width, 3), np.uint8)
    img[measured & ~ok] = BEYOND_C
    img[ok] = _depth_lut()[(v[ok] * 255).astype(np.uint8), 0]
    return img


def _depth_lut() -> np.ndarray:
    return cv2.applyColorMap(np.arange(256, dtype=np.uint8)[::-1, None], cv2.COLORMAP_TURBO)


def depth_scale_bar(width: int, d_min: float, d_max: float, ticks) -> np.ndarray:
    """The colour scale in metres, as a strip to put under the pictures."""
    bar = np.full((34, width, 3), 24, np.uint8)
    x0, x1 = 10, width - 10
    bar[4:14, x0:x1] = _depth_lut()[np.linspace(0, 255, x1 - x0).astype(np.uint8), 0][None]
    for k, t in enumerate(ticks):
        px = round(x0 + (t - d_min) / (d_max - d_min) * (x1 - x0 - 1))
        bar[14:18, px] = 255
        text = f'{t:g}' + (' m' if k == len(ticks) - 1 else '')
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        tx = min(max(px - size // 2, 2), width - size - 2)
        cv2.putText(
            bar, text, (tx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA
        )
    return bar


def _caption(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (8 + 7 * len(text), 18), (24, 24, 24), -1)
    cv2.putText(img, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1, cv2.LINE_AA)


def render_depth_pair(
    left: np.ndarray,
    right: np.ndarray,
    d_min: float,
    d_max: float,
    captions: tuple[str, str],
    ticks,
    width: int = 320,
) -> np.ndarray:
    """Two depth images of the same view side by side, on one colour scale, scale bar below."""
    a = colour_depth(left, d_min, d_max, width)
    b = colour_depth(right, d_min, d_max, width)
    _caption(a, captions[0])
    _caption(b, captions[1])
    gap = np.full((a.shape[0], 4, 3), 24, np.uint8)
    top = np.hstack([a, gap, b])
    return np.vstack([top, depth_scale_bar(top.shape[1], d_min, d_max, ticks)])


def render_depth_single(depth, d_min, d_max, caption, ticks, width: int = 320) -> np.ndarray:
    img = colour_depth(depth, d_min, d_max, width)
    _caption(img, caption)
    return np.vstack([img, depth_scale_bar(width, d_min, d_max, ticks)])


def encode_png(img: np.ndarray, colours: int = 64) -> bytes:
    """PNG of a BGR picture. With Pillow, as a palette PNG of `colours` colours (no dithering):
    the maps have few colours, and this halves their size for a slow link."""
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None and colours:
        pal = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1])).quantize(
            colors=colours, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE
        )
        out = io.BytesIO()
        pal.save(out, 'PNG', compress_level=6)
        return out.getvalue()
    ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError('PNG encoding failed')
    return buf.tobytes()
