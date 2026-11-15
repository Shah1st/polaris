"""See the rover's D405 depth camera from the rover page, through its capture server.

`rover_mapper/d405_server.py` owns the RealSense camera on the robot (one program per camera,
like the serial port): this module only asks it for pictures over HTTP (127.0.0.1:8091 by
default) and makes them viewable:

* the colour picture, as the capture server gives it;
* the depth (16-bit raw units, in the depth camera's frame, not aligned with the colour) as a
  colour map: near = red, far = blue, no measure = black;
* a few numbers for a quick check: distance in the centre of the picture, nearest point,
  share of pixels with a measure, age of the picture.

The D405 sees from about 7 cm; it is made for short range (best under about 50 cm).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np

try:
    import cv2
except ImportError:  # the rover server runs without OpenCV; then there is no D405 view
    cv2 = None

DEFAULT_URL = 'http://127.0.0.1:8091'
NEAREST_M = 0.07  # the D405 measures nothing closer


class D405UnavailableError(Exception):
    """The capture server does not answer, or has no picture."""


class D405View:
    def __init__(self, url: str | None = DEFAULT_URL, timeout: float = 1.5, opener=None) -> None:
        self.url = url.rstrip('/') if url else None
        self.timeout = timeout
        self._open = opener or urllib.request.urlopen
        self._scale: float | None = None  # metres per raw depth unit, from /info

    @property
    def enabled(self) -> bool:
        return self.url is not None

    def _get(self, path: str) -> tuple[bytes, dict]:
        if not self.enabled:
            raise D405UnavailableError('D405 view disabled (--d405 off)')
        try:
            with self._open(self.url + path, timeout=self.timeout) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get('error', '')
            except (ValueError, AttributeError):
                detail = ''
            raise D405UnavailableError(f'D405 server: HTTP {e.code} {detail}'.strip()) from e
        except (urllib.error.URLError, OSError) as e:
            raise D405UnavailableError(f'D405 server not reachable at {self.url} ({e})') from e

    def _json(self, path: str) -> dict:
        body, _ = self._get(path)
        return json.loads(body)

    def info(self) -> dict:
        info = self._json('/info')
        self._scale = float(info['depth_scale_m'])
        return info

    def color_jpg(self) -> tuple[bytes, str | None]:
        """(JPEG bytes, age in ms from the capture server)."""
        body, headers = self._get('/color.jpg')
        return body, headers.get('X-Age-Ms')

    def depth_m(self) -> tuple[np.ndarray, str | None]:
        """Depth in metres (float32, 0 = no measure), and its age in ms."""
        if cv2 is None:
            raise D405UnavailableError('OpenCV is missing on this computer')
        if self._scale is None:
            self.info()
        body, headers = self._get('/depth.png')
        raw = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            raise D405UnavailableError('D405 server: depth is not a 16-bit PNG')
        return raw.astype(np.float32) * np.float32(self._scale), headers.get('X-Age-Ms')

    def depth_jpg(self, max_m: float = 1.0) -> tuple[bytes, str | None]:
        depth, age = self.depth_m()
        return colorize(depth, max_m), age

    def status(self) -> dict:
        """Capture server health, camera identity, and numbers on the latest depth."""
        out: dict = {'url': self.url}
        try:
            out['health'] = self._json('/health')
            info = self.info()
            out['camera'] = {k: info.get(k) for k in ('name', 'serial', 'firmware', 'usb', 'mode')}
            depth, age = self.depth_m()
            out['depth'] = {**depth_stats(depth), 'age_ms': None if age is None else int(age)}
            out['ok'] = bool(out['health'].get('ok'))
        except D405UnavailableError as e:
            out['ok'] = False
            out['error'] = str(e)
        return out


def depth_stats(depth: np.ndarray) -> dict:
    """Distance in the centre (median of the middle 20 % box), nearest point (2nd percentile,
    so that a few noisy pixels do not count), share of pixels with a measure."""
    valid = depth > 0
    h, w = depth.shape
    centre = depth[int(0.4 * h) : int(0.6 * h), int(0.4 * w) : int(0.6 * w)]
    centre = centre[centre > 0]
    return {
        'centre_m': round(float(np.median(centre)), 3) if centre.size else None,
        'nearest_m': round(float(np.percentile(depth[valid], 2)), 3) if valid.any() else None,
        'valid': round(float(valid.mean()), 3),
        'width': int(w),
        'height': int(h),
    }


def colorize(depth: np.ndarray, max_m: float = 1.0) -> bytes:
    """JPEG colour map of a depth image in metres: red near, blue at max_m and beyond."""
    max_m = max(max_m, NEAREST_M + 0.05)
    valid = depth > 0
    near = np.clip((depth - NEAREST_M) / (max_m - NEAREST_M), 0.0, 1.0)
    img = cv2.applyColorMap((255 * (1.0 - near)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0
    cv2.putText(img, f'{NEAREST_M:.2f}-{max_m:.1f} m', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)  # fmt: skip
    ok, jpg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise D405UnavailableError('could not encode the depth picture')
    return jpg.tobytes()
