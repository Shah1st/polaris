"""The air camera as a measuring instrument: from a picture and the pose the phone gives with
it (ARKit, see README.md), to metres on the floor. No ROS, numpy and OpenCV only.

Floor frame (the one frame of the whole chain): metres, x and y on the floor, z up, z = 0 on
the floor. It is ARKit's world frame, which stays put while the phone moves, turned so that z
is up: x_floor = x_arkit, y_floor = -z_arkit, z_floor = y_arkit - floor_y. Its origin is under
the place where the phone was when the app started tracking; its x axis is arbitrary but fixed.

    pose = Pose.from_arkit(header)            # the X-Pose header of one picture
    xy = pose.pixel_to_floor([[640, 360]])    # where that pixel is on the floor, in metres
    top = pose.floor_view(image, (-1, -1, 1, 1), 0.005)   # the same top view whatever the pose
    rover = pose.locate_marker(corners, 0.12) # a marker 12 cm above the floor: x, y, heading
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# ARKit world (x, y up, z towards the viewer) -> floor frame (x, y, z up): a rotation.
_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
# ARKit camera (x right, y up, looks along -z) -> picture camera (x right, y down, looks along z)
_CAMERA = np.diag([1.0, -1.0, -1.0])


class PoseError(ValueError):
    """The pose of a picture is missing, incomplete, or cannot see the floor."""


@dataclass(frozen=True)
class Marker:
    x: float  # centre, metres in the floor frame
    y: float
    heading: float  # radians, direction of the marker's top edge (corner 0 -> corner 1)
    side: float  # measured side, metres: compare with the printed size to trust the pose


@dataclass(frozen=True)
class Pose:
    """Where the camera was when it took one picture."""

    t: float  # the phone's clock, seconds
    width: int
    height: int
    k: np.ndarray  # 3x3 intrinsics, pixels of that picture
    r: np.ndarray  # 3x3, picture camera axes (x right, y down, z forward) in the floor frame
    c: np.ndarray  # camera centre in the floor frame; c[2] is its height above the floor
    tracking: str = 'normal'  # 'normal', 'limited:<why>' or 'none'
    floor_known: bool = True  # False: no floor found yet, z = 0 is where the phone started
    session: int = 0  # changes when the app restarts its tracking: the floor frame is a new one

    @classmethod
    def from_arkit(cls, d: dict, floor_y: float | None = None) -> Pose:
        """From the X-Pose header: {t, sid, w, h, k: [fx, fy, cx, cy], T: 16 numbers (camera
        to world, row after row), track, floor_y}. `floor_y` overrides the header's."""
        try:
            fx, fy, cx, cy = (float(v) for v in d['k'])
            t44 = np.array(d['T'], dtype=float).reshape(4, 4)
            width, height = int(d['w']), int(d['h'])
        except (KeyError, TypeError, ValueError) as e:
            raise PoseError(f'incomplete pose: {e}') from e
        if not np.isfinite(t44).all() or min(fx, fy) <= 0:
            raise PoseError('pose with non-finite numbers or no focal length')
        if floor_y is None:
            floor_y = d.get('floor_y')
        c = _WORLD @ t44[:3, 3]
        c[2] -= floor_y if floor_y is not None else 0.0
        return cls(
            t=float(d.get('t', 0.0)),
            width=width,
            height=height,
            k=np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]),
            r=_WORLD @ t44[:3, :3] @ _CAMERA,
            c=c,
            tracking=str(d.get('track', 'normal')),
            floor_known=floor_y is not None,
            session=int(d.get('sid', 0)),
        )

    # -- geometry -------------------------------------------------------------------------

    def project(self, points) -> np.ndarray:
        """Floor-frame points (N x 3, metres) -> pixels (N x 2). NaN behind the camera."""
        p = (np.atleast_2d(np.asarray(points, dtype=float)) - self.c) @ self.r
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = (p @ self.k.T)[:, :2] / p[:, 2:3]
        uv[p[:, 2] <= 1e-6] = np.nan
        return uv

    def plane_homography(self, z: float = 0.0) -> np.ndarray:
        """3x3 matrix from (x, y, 1) on the horizontal plane at height z to pixels."""
        rt = self.r.T
        return self.k @ np.column_stack([rt[:, 0], rt[:, 1], rt @ (np.array([0, 0, z]) - self.c)])

    def pixel_to_floor(self, pixels, z: float = 0.0) -> np.ndarray:
        """Pixels (N x 2) -> (x, y) in metres where their rays meet the plane at height z.
        NaN for a pixel that looks at or above the horizon."""
        uv = np.atleast_2d(np.asarray(pixels, dtype=float))
        rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(self.k).T @ self.r.T
        with np.errstate(divide='ignore', invalid='ignore'):
            s = (z - self.c[2]) / rays[:, 2]
        xy = self.c[:2] + s[:, None] * rays[:, :2]
        xy[~(s > 0)] = np.nan  # the plane is behind the camera along that ray
        return xy

    def footprint(self, z: float = 0.0) -> np.ndarray:
        """The four corners of the picture on the floor (4 x 2, NaN above the horizon)."""
        w, h = self.width - 1, self.height - 1
        return self.pixel_to_floor([[0, 0], [w, 0], [w, h], [0, h]], z)

    @property
    def tilt_deg(self) -> float:
        """Angle between the optical axis and straight down: 0 = looking straight down."""
        return math.degrees(math.acos(max(-1.0, min(1.0, -self.r[2, 2]))))

    # -- pictures -------------------------------------------------------------------------

    def floor_view(self, image, bounds, res: float = 0.005, z: float = 0.0, fill=0):
        """The picture laid flat on the floor: a top view of `bounds` = (x0, y0, x1, y1) in
        metres, `res` metres per pixel, x to the right and y up. A point of the floor lands
        on the same pixel whatever the camera's pose; what stands above the floor leans away
        from the point under the camera. Outside the picture: `fill`."""
        x0, y0, x1, y1 = bounds
        cols, rows = round((x1 - x0) / res), round((y1 - y0) / res)
        if cols <= 0 or rows <= 0:
            raise PoseError('empty bounds')
        if self.c[2] - z <= 0.05:
            raise PoseError('the camera is not above that plane')
        to_floor = np.array([[res, 0.0, x0 + res / 2], [0.0, -res, y1 - res / 2], [0, 0, 1.0]])
        return cv2.warpPerspective(
            image,
            self.plane_homography(z) @ to_floor,
            (cols, rows),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill,
        )

    def seen_mask(self, bounds, res: float = 0.005, z: float = 0.0) -> np.ndarray:
        """Which pixels of `floor_view` are inside the picture (bool, same size)."""
        ones = np.full((self.height, self.width), 255, np.uint8)
        return self.floor_view(ones, bounds, res, z) > 127

    # -- the rover ------------------------------------------------------------------------

    def locate_marker(self, corners, height: float) -> Marker:
        """A square marker lying flat `height` metres above the floor (the rover's roof), from
        its four corners in the picture, in ArUco's order (top-left, top-right, bottom-right,
        bottom-left of the printed marker)."""
        p = self.pixel_to_floor(np.asarray(corners, dtype=float).reshape(4, 2), height)
        if not np.isfinite(p).all():
            raise PoseError('marker above the horizon')
        top = p[1] - p[0]
        sides = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
        x, y = p.mean(axis=0)
        return Marker(float(x), float(y), math.atan2(top[1], top[0]), float(np.mean(sides)))
