"""Write printable ArUco marker PNGs.

Usage: ros2 run guidance_calib generate_markers --ids 0 1 2 3 10 --out markers
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from guidance_calib.aruco_compat import DEFAULT_DICTIONARY, generate_marker


def marker_with_border(marker_id: int, side_px: int, dictionary_name: str) -> np.ndarray:
    """Return the marker with the white quiet zone the detector needs around it."""
    marker = generate_marker(marker_id, side_px, dictionary_name)
    border = side_px // 4
    return cv2.copyMakeBorder(
        marker, border, border, border, border, cv2.BORDER_CONSTANT, value=255
    )


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', type=int, nargs='+', default=[0, 1, 2, 3, 10])
    parser.add_argument('--out', type=Path, default=Path('markers'))
    parser.add_argument('--side-px', type=int, default=800)
    parser.add_argument('--dictionary', default=DEFAULT_DICTIONARY)
    options = parser.parse_args(args)

    options.out.mkdir(parents=True, exist_ok=True)
    for marker_id in options.ids:
        path = options.out / f'aruco_{marker_id}.png'
        image = marker_with_border(marker_id, options.side_px, options.dictionary)
        cv2.imwrite(str(path), image)
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
