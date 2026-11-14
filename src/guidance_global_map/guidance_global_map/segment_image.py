"""Run the overhead pipeline on one image file. No ROS needed.

For tuning on photos taken straight down from a known height, without floor markers. The
image must be at the camera's native scale (cropping is fine, resizing is not).

    python -m guidance_global_map.segment_image photo.jpg out/scene --camera-height 1.9

It writes, with the given prefix:

- PREFIX_overlay.jpg: the view from above with obstacles red, the rover blue, margin cells
  orange (only with --margin), and every blob labelled with its size in centimetres;
- PREFIX_grid.png: the occupancy grid the node would publish on /map, drawn cell by cell
  (white free, grey obstacle, blue rover, black margin, light blue unknown);
- PREFIX_grid.csv and PREFIX_grid.npy: the same grid as numbers, one per cell (0 free,
  100 obstacle, 50 rover, 99 margin, -1 unknown), row 0 at y = 0;
- PREFIX_fusion.npy: the grid in the convention of the map fusion in merging_map
  (1 obstacle, 0 free, -1 not seen), ready for its decouper() at level z = 4 when the
  cell size is 5 cm.

With --rover-marker ID the rover is the ArUco marker with that id on its top, which also
gives its heading; without it, the rover is the blob of the rover's size.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from guidance_global_map.pipeline import OverheadResult, OverheadSettings, process_image
from guidance_global_map.segmentation import MARGIN, grid_to_fusion, grid_to_image, margin_cells


def tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    image[mask] = (0.45 * image[mask] + 0.55 * np.array(color, dtype=np.float32)).astype(np.uint8)


def draw_overlay(result: OverheadResult, factor: int) -> np.ndarray:
    """Return the top-down view with margin cells orange, obstacles red and the rover blue."""
    overlay = result.top_down.copy()
    margin = np.flipud(np.kron(result.grid == MARGIN, np.ones((factor, factor), dtype=bool)))
    rows, cols = overlay.shape[:2]
    tint(overlay, margin[-rows:, :cols], (0, 140, 255))
    for blob in result.blobs:
        tint(overlay, blob.mask, (0, 0, 255))
    if result.rover_mask is not None:
        overlay[result.rover_mask] = result.top_down[result.rover_mask]
        tint(overlay, result.rover_mask, (255, 80, 0))
    scale = max(1, round(900 / max(overlay.shape[:2])))
    overlay = cv2.resize(overlay, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for blob in result.blobs:
        label = f'{blob.size[0] * 100:.0f}x{blob.size[1] * 100:.0f}'
        if blob.mask is result.rover_mask:
            label = 'rover ' + label
        origin = (int(blob.center[0] * scale) - 30, int(blob.center[1] * scale))
        cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(overlay, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return overlay


def main(args: list[str] | None = None) -> None:
    defaults = OverheadSettings()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('image')
    parser.add_argument('prefix', type=Path, help='output files are PREFIX_overlay.jpg etc.')
    parser.add_argument('--metres-per-pixel', type=float, help='measured scale; overrides lens')
    parser.add_argument('--camera-height', type=float, default=defaults.camera_height)
    parser.add_argument('--focal-length-35mm', type=float, default=defaults.focal_length_35mm)
    parser.add_argument('--sensor-size', type=int, nargs=2, default=list(defaults.sensor_size))
    parser.add_argument('--resolution', type=float, default=defaults.resolution)
    parser.add_argument('--cell-size', type=float, default=defaults.cell_size)
    parser.add_argument('--threshold', type=float, default=defaults.color_threshold)
    parser.add_argument('--lightness-weight', type=float, default=defaults.lightness_weight)
    parser.add_argument('--floor-window', type=float, default=defaults.floor_window)
    parser.add_argument('--min-area', type=float, default=defaults.min_obstacle_area)
    parser.add_argument('--margin', type=float, default=defaults.margin)
    parser.add_argument('--rover-size', type=float, nargs=2, default=list(defaults.rover_size))
    parser.add_argument('--rover-tolerance', type=float, default=defaults.rover_size_tolerance)
    parser.add_argument('--rover-marker', type=int, help='ArUco id on the rover (DICT_4X4_50)')
    options = parser.parse_args(args)

    image = cv2.imread(options.image)
    if image is None:
        raise SystemExit(f'cannot read {options.image}')
    settings = OverheadSettings(
        metres_per_pixel=options.metres_per_pixel,
        camera_height=options.camera_height,
        focal_length_35mm=options.focal_length_35mm,
        sensor_size=tuple(options.sensor_size),
        resolution=options.resolution,
        color_threshold=options.threshold,
        lightness_weight=options.lightness_weight,
        floor_window=options.floor_window,
        min_obstacle_area=options.min_area,
        cell_size=options.cell_size,
        margin=options.margin,
        rover_marker_id=options.rover_marker,
        rover_size=tuple(options.rover_size),
        rover_size_tolerance=options.rover_tolerance,
    )
    result = process_image(image, settings)
    grid = result.grid
    factor = int(round(settings.cell_size / settings.resolution))

    options.prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = str(options.prefix)
    cv2.imwrite(f'{stem}_overlay.jpg', draw_overlay(result, factor))
    cv2.imwrite(f'{stem}_grid.png', grid_to_image(grid))
    np.save(f'{stem}_grid.npy', grid)
    np.savetxt(f'{stem}_grid.csv', grid, fmt='%d', delimiter=',')
    np.save(f'{stem}_fusion.npy', grid_to_fusion(grid))

    bounds = result.bounds
    print(f'area {bounds.x_max:.2f} x {bounds.y_max:.2f} m, {len(result.blobs)} blobs')
    if result.rover_position is None:
        print('rover: not found')
    else:
        x, y = result.rover_position
        heading = result.rover_heading
        facing = 'unknown' if heading is None else f'{math.degrees(heading):.0f} deg'
        print(f'rover: x={x:.2f} y={y:.2f} m, heading {facing}')
    reach = margin_cells(settings.margin, settings.cell_size)
    print(
        f'grid: {grid.shape[1]} x {grid.shape[0]} cells of {settings.cell_size * 100:g} cm, '
        f'margin {reach} cells = {reach * settings.cell_size * 100:g} cm'
    )
    print(f'wrote {stem}_overlay.jpg, {stem}_grid.png, {stem}_grid.csv, {stem}_grid.npy')


if __name__ == '__main__':
    main()
