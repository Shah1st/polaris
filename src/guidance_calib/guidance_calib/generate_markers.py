"""Write printable ArUco marker PNGs.

Usage: ros2 run guidance_calib generate_markers --ids 0 1 2 3 10 --out markers
       python -m guidance_calib.generate_markers --ids 10 --side-cm 12 --out markers
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


def _set_print_size(path: Path, pixels_per_cm: float) -> None:
    """Write the PNG's physical size (its pHYs chunk), so that "print at 100 %" works."""
    import struct
    import zlib

    data = path.read_bytes()
    per_metre = int(round(pixels_per_cm * 100))
    body = b'pHYs' + struct.pack('>IIB', per_metre, per_metre, 1)
    chunk = struct.pack('>I', 9) + body + struct.pack('>I', zlib.crc32(body) & 0xFFFFFFFF)
    end_of_header = 8 + 4 + 4 + 13 + 4  # signature, then the IHDR chunk
    path.write_bytes(data[:end_of_header] + chunk + data[end_of_header:])


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', type=int, nargs='+', default=[0, 1, 2, 3, 10])
    parser.add_argument('--out', type=Path, default=Path('markers'))
    parser.add_argument('--side-px', type=int, default=800)
    parser.add_argument(
        '--side-cm',
        type=float,
        help='side of the black square once printed (cm): the PNG then carries its print '
        'size, so printing at 100 %% gives that size',
    )
    parser.add_argument('--dictionary', default=DEFAULT_DICTIONARY)
    options = parser.parse_args(args)

    options.out.mkdir(parents=True, exist_ok=True)
    for marker_id in options.ids:
        path = options.out / f'aruco_{marker_id}.png'
        image = marker_with_border(marker_id, options.side_px, options.dictionary)
        cv2.imwrite(str(path), image)
        if options.side_cm:
            _set_print_size(path, options.side_px / options.side_cm)
            total = options.side_cm * image.shape[0] / options.side_px
            print(
                f'wrote {path}: black square {options.side_cm:g} cm, {total:.1f} cm with its '
                'white border, when printed at 100 %'
            )
        else:
            print(f'wrote {path}')


if __name__ == '__main__':
    main()
