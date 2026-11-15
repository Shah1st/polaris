"""A small synthetic world for the tests: a textured floor, a rover marker above it, and a
camera that says its pose the way the iPhone app does (ARKit conventions)."""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

FLOOR_BOUNDS = (-1.5, -1.5, 1.5, 1.5)  # metres
FLOOR_RES = 0.002  # metres per pixel of the floor texture
FLOOR_Y = -1.37  # ARKit world height of the floor: the phone started 1.37 m above it
K = (1000.0, 1000.0, 640.0, 360.0)
SIZE = (1280, 720)


def floor_texture(seed: int = 3) -> np.ndarray:
    """A grey floor with coloured boxes and speckles: enough texture to compare two views."""
    rng = np.random.default_rng(seed)
    n = round((FLOOR_BOUNDS[2] - FLOOR_BOUNDS[0]) / FLOOR_RES)
    img = np.full((n, n, 3), 150, np.uint8)
    img += rng.integers(0, 12, img.shape, dtype=np.uint8)
    for _ in range(25):
        x, y = rng.integers(100, n - 100, 2)
        w, h = rng.integers(30, 120, 2)
        colour = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, -1)
    return img


def arkit_pose(eye, target, roll_deg: float = 0.0, t: float = 0.0, floor_y=FLOOR_Y) -> dict:
    """The X-Pose header of a camera at `eye` looking at `target` (both in the floor frame,
    metres), as ARKit would give it: camera-to-world, y up, camera looking along its -z."""

    def to_arkit(p):  # floor frame (x, y, z up) -> ARKit world (x, y up, z)
        x, y, z = p
        return np.array([x, z + (floor_y or 0.0), -y])

    e, g = to_arkit(eye), to_arkit(target)
    f = (g - e) / np.linalg.norm(g - e)
    hint = np.array([0.0, 0.0, -1.0]) if abs(f[1]) > 0.99 else np.array([0.0, 1.0, 0.0])
    x = np.cross(f, hint)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    a = math.radians(roll_deg)
    x, y = math.cos(a) * x + math.sin(a) * y, -math.sin(a) * x + math.cos(a) * y
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, -f, e
    return {
        't': t,
        'w': SIZE[0],
        'h': SIZE[1],
        'k': list(K),
        'T': [float(v) for v in m.ravel()],
        'track': 'normal',
        'floor_y': floor_y,
    }


def arkit_project(d: dict, point) -> tuple[float, float]:
    """A floor-frame point to a pixel, written from Apple's description of ARCamera alone
    (independent of pose_camera.py): world -> camera with the inverse transform, then the
    intrinsics with the picture's y pointing down and the camera looking along -z."""
    x, y, z = point
    world = np.array([x, z + (d['floor_y'] or 0.0), -y, 1.0])
    cam = np.linalg.inv(np.array(d['T']).reshape(4, 4)) @ world
    fx, fy, cx, cy = d['k']
    return fx * cam[0] / -cam[2] + cx, fy * -cam[1] / -cam[2] + cy


def render(pose, floor: np.ndarray, marker=None, patches=()) -> np.ndarray:
    """The picture a camera at `pose` (a pose_camera.Pose) takes of the floor, and of flat
    things lying above it: marker = (image, x, y, top_edge_angle_rad, side_m, height_m), and
    `patches` of the same form (box tops; their sides are not drawn). Higher things are drawn
    over lower ones."""
    x0, _, _, y1 = FLOOR_BOUNDS
    tex_to_floor = np.array([[FLOOR_RES, 0, x0], [0, -FLOOR_RES, y1], [0, 0, 1.0]])
    out = cv2.warpPerspective(floor, pose.plane_homography(0.0) @ tex_to_floor, SIZE)
    flat = list(patches) + ([marker] if marker is not None else [])
    for img, mx, my, angle, side, height in sorted(flat, key=lambda f: f[5]):
        s = side / img.shape[0]
        c, sn = math.cos(angle), math.sin(angle)
        half = img.shape[0] / 2
        # pixel (u right, v down) -> its own axes (right, up) -> the floor frame
        local = np.array([[s, 0, -half * s], [0, -s, half * s], [0, 0, 1.0]])
        place = np.array([[c, -sn, mx], [sn, c, my], [0, 0, 1.0]])
        h = pose.plane_homography(height) @ place @ local
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        warped = cv2.warpPerspective(img, h, SIZE)
        mask = cv2.warpPerspective(np.full(img.shape[:2], 255, np.uint8), h, SIZE) > 127
        out[mask] = warped[mask]
    return out


def box_top(seed: int, px: int = 200) -> np.ndarray:
    """The top of a box: cardboard with print on it (some texture, as real boxes have)."""
    rng = np.random.default_rng(seed)
    img = np.full((px, px, 3), (90, 140, 180), np.uint8)
    img += rng.integers(0, 25, img.shape, dtype=np.uint8)
    for _ in range(12):
        x, y = rng.integers(0, px - 20, 2)
        cv2.putText(img, 'POLARIS', (int(x), int(y) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    tuple(int(v) for v in rng.integers(0, 120, 3)), 1)  # fmt: skip
    return img


def marker_image(marker_id: int = 10, px: int = 240, family: str = 'DICT_APRILTAG_25h9'):
    """The printed marker (the team's AprilTag 25h9 by default) with its white margin, a
    quarter of the side all around: its black square is 2/3 of the picture's side."""
    code = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, family)), marker_id, px
    )
    m = px // 4
    return cv2.copyMakeBorder(code, m, m, m, m, cv2.BORDER_CONSTANT, value=255)


def header(d: dict) -> str:
    return json.dumps(d, separators=(',', ':'))
