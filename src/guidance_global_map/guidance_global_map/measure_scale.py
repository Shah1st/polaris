"""Measure the overhead camera's scale (`metres_per_pixel`) from two floor points. No ROS.

Lay a tape measure or a ruler flat on the floor, in the middle of the view, and note two marks
a known distance apart (as long as possible: 1 m or more). Then either give their pixels:

    python -m guidance_global_map.measure_scale frame.jpg --points 412 530 1310 541 --distance 1.0

or click them in a window (needs a display, so on a laptop, not on the VM):

    python -m guidance_global_map.measure_scale frame.jpg --distance 1.0 --click

The image can also be a live source as `camera_reader` takes it (camera index, http or rtsp
URL); `--save frame.jpg` keeps the frame used, to find the pixels in any image viewer.
Measure on frames of the size the mission uses: the scale is per pixel of that size.
"""

import argparse
import math

import cv2
import numpy as np

from guidance_global_map.camera_source import camera_reader


def metres_per_pixel_from_points(
    first: tuple[float, float], second: tuple[float, float], distance: float
) -> float:
    """Ground size of one pixel, from two image points `distance` metres apart on the floor."""
    pixels = math.dist(first, second)
    if pixels <= 0 or distance <= 0:
        raise ValueError('need two different points and a positive distance')
    return distance / pixels


def grab_frame(source: str, timeout: float = 10.0) -> np.ndarray | None:
    """One frame of a file or a live source."""
    read = camera_reader(source, max_age=timeout)
    try:
        return read.read_new(timeout)
    finally:
        read.close()


def click_two_points(image: np.ndarray) -> list[tuple[float, float]]:
    """Let the user click two points in a window; returns them in image pixels."""
    scale = min(1.0, 1200 / max(image.shape[:2]))
    shown = cv2.resize(image, None, fx=scale, fy=scale)
    points: list[tuple[float, float]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x / scale, y / scale))
            cv2.circle(shown, (x, y), 5, (0, 0, 255), -1)

    cv2.namedWindow('measure_scale')
    cv2.setMouseCallback('measure_scale', on_mouse)
    while len(points) < 2:
        cv2.imshow('measure_scale', shown)
        if cv2.waitKey(30) == 27:  # Esc
            break
    cv2.destroyAllWindows()
    return points


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('image', help='image file, camera index, or http/rtsp URL')
    parser.add_argument('--distance', type=float, required=True, help='metres between points')
    parser.add_argument('--points', type=float, nargs=4, metavar=('U1', 'V1', 'U2', 'V2'))
    parser.add_argument('--click', action='store_true', help='click the two points')
    parser.add_argument('--save', help='write the frame used to this file')
    options = parser.parse_args(args)

    image = grab_frame(options.image)
    if image is None:
        raise SystemExit(f'no image from {options.image}')
    if options.save:
        cv2.imwrite(options.save, image)
    rows, cols = image.shape[:2]
    if options.points:
        u1, v1, u2, v2 = options.points
        points = [(u1, v1), (u2, v2)]
    elif options.click:
        points = click_two_points(image)
        if len(points) < 2:
            raise SystemExit('cancelled')
    else:
        raise SystemExit(f'image is {cols}x{rows}; give --points U1 V1 U2 V2 or --click')
    scale = metres_per_pixel_from_points(points[0], points[1], options.distance)
    print(f'image {cols}x{rows}, {math.dist(*points):.1f} px for {options.distance} m')
    print(f'metres_per_pixel: {scale:.6f}  (view {cols * scale:.2f} x {rows * scale:.2f} m)')


if __name__ == '__main__':
    main()
