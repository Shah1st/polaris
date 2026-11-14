"""Run the overhead pipeline on one image file. No ROS needed.

For tuning on photos taken straight down from a known height, without floor markers. The
image must be at the camera's native scale (cropping is fine, resizing is not).

    python -m guidance_global_map.segment_image photo.jpg out/scene --camera-height 1.9

It writes, with the given prefix:

- PREFIX_overlay.jpg: the view from above with obstacles red, the rover blue, margin cells
  orange (only with --margin), and every blob labelled with its size in centimetres;
- PREFIX_grid.png: the occupancy grid the node would publish on /map, drawn cell by cell
  (white free, grey obstacle, blue rover, black margin, light blue unknown), except that
  the published map leaves the rover's cells free;
- PREFIX_grid.csv and PREFIX_grid.npy: the same grid as numbers, one per cell (0 free,
  100 obstacle, 50 rover, 99 margin, -1 unknown), row 0 at y = 0.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from guidance_calib.ground_plane import (
    MapBounds,
    metres_per_pixel_from_height,
    straight_down_homography,
    warp_to_top_down,
)
from guidance_global_map.segmentation import (
    MARGIN,
    Blob,
    build_cell_grid,
    find_blobs,
    find_rover_blob,
    grid_to_image,
    margin_cells,
    segment_obstacles,
)


def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    image[mask] = (0.45 * image[mask] + 0.55 * np.array(color, dtype=np.float32)).astype(np.uint8)


def draw_overlay(
    top_down: np.ndarray, blobs: list[Blob], rover: Blob | None, grid: np.ndarray, factor: int
) -> np.ndarray:
    """Return the top-down view with margin cells orange, obstacles red and the rover blue."""
    overlay = top_down.copy()
    margin = np.flipud(np.kron(grid == MARGIN, np.ones((factor, factor), dtype=bool)))
    rows, cols = overlay.shape[:2]
    tint(overlay, margin[-rows:, :cols], (0, 140, 255))
    for blob in blobs:
        tint(overlay, blob.mask, (255, 80, 0) if blob is rover else (0, 0, 255))
    scale = max(1, round(900 / max(overlay.shape[:2])))
    overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for blob in blobs:
        label = f'{blob.size[0] * 100:.0f}x{blob.size[1] * 100:.0f}'
        if blob is rover:
            label = 'rover ' + label
        origin = (int(blob.center[0] * scale) - 30, int(blob.center[1] * scale))
        cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return overlay


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('image')
    parser.add_argument('prefix', type=Path, help='output files are PREFIX_overlay.jpg etc.')
    parser.add_argument('--camera-height', type=float, default=1.9, help='metres')
    parser.add_argument('--focal-length-35mm', type=float, default=26.0)
    parser.add_argument('--sensor-size', type=int, nargs=2, default=[4032, 3024], help='pixels')
    parser.add_argument('--resolution', type=float, default=0.005, help='detection, metres')
    parser.add_argument('--cell-size', type=float, default=0.025, help='grid cell, metres')
    parser.add_argument('--threshold', type=float, default=25.0)
    parser.add_argument('--lightness-weight', type=float, default=0.7)
    parser.add_argument('--floor-window', type=float, default=1.0, help='metres; 0 = global')
    parser.add_argument('--min-area', type=float, default=0.0005, help='square metres')
    parser.add_argument('--rover-size', type=float, nargs=2, default=[0.20, 0.23], help='metres')
    parser.add_argument('--rover-tolerance', type=float, default=0.2)
    parser.add_argument('--margin', type=float, default=0.0, help='metres around obstacles')
    options = parser.parse_args(args)

    image = cv2.imread(options.image)
    if image is None:
        raise SystemExit(f'cannot read {options.image}')
    metres_per_pixel = metres_per_pixel_from_height(
        options.camera_height, options.focal_length_35mm, *options.sensor_size
    )
    rows, cols = image.shape[:2]
    homography = straight_down_homography(metres_per_pixel, rows)
    bounds = MapBounds(0.0, 0.0, cols * metres_per_pixel, rows * metres_per_pixel)
    top_down, valid = warp_to_top_down(image, homography, bounds, options.resolution)
    obstacles = segment_obstacles(
        top_down,
        valid,
        color_threshold=options.threshold,
        lightness_weight=options.lightness_weight,
        min_area_px=int(round(options.min_area / options.resolution**2)),
        floor_window_px=int(round(options.floor_window / options.resolution)),
    )
    blobs = find_blobs(obstacles, options.resolution)
    rover = find_rover_blob(blobs, tuple(options.rover_size), options.rover_tolerance)
    if rover is not None:
        obstacles = obstacles & ~rover.mask
    margin = options.margin
    grid = build_cell_grid(
        obstacles,
        valid,
        options.resolution,
        options.cell_size,
        margin,
        rover=None if rover is None else rover.mask,
    )
    factor = int(round(options.cell_size / options.resolution))

    options.prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = str(options.prefix)
    cv2.imwrite(f'{stem}_overlay.jpg', draw_overlay(top_down, blobs, rover, grid, factor))
    cv2.imwrite(f'{stem}_grid.png', grid_to_image(grid))
    np.save(f'{stem}_grid.npy', grid)
    np.savetxt(f'{stem}_grid.csv', grid, fmt='%d', delimiter=',')

    print(f'area {bounds.x_max:.2f} x {bounds.y_max:.2f} m, {len(blobs)} blobs')
    if rover is None:
        print('rover: not found')
    else:
        x = rover.center[0] * options.resolution
        y = bounds.y_max - rover.center[1] * options.resolution
        print(f'rover: x={x:.2f} y={y:.2f} m, seen as {rover.size[0]:.2f} x {rover.size[1]:.2f} m')
    reach = margin_cells(margin, options.cell_size)
    print(
        f'grid: {grid.shape[1]} x {grid.shape[0]} cells of {options.cell_size * 100:g} cm, '
        f'margin {reach} cells = {reach * options.cell_size * 100:g} cm'
    )
    print(f'wrote {stem}_overlay.jpg, {stem}_grid.png, {stem}_grid.csv, {stem}_grid.npy')


if __name__ == '__main__':
    main()
