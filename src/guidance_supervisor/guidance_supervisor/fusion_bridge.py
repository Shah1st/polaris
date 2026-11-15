"""Between the ROS topics and the map fusion (`merging_map/`). No ROS import, so it can be
tested with plain pytest; `merging_map_node.py` only subscribes, calls this and publishes.

- `/map` (nav_msgs/OccupancyGrid from the overhead camera) becomes drone tiles;
- `/rover/pose_overhead` gives the rover position, and a heading only when it is known;
- the merged map goes back out as an OccupancyGrid.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np

# Cell values of the overhead camera's /map (docs/overhead-camera.md, branch overhead-camera)
OCCUPIED = 100
ROVER = 50  # the rover itself: free for the fusion, which takes the rover position apart
MARGIN = 99  # margin around obstacles: free for the fusion, which grows obstacles itself
UNKNOWN = -1
UNKNOWN_HEADING_VARIANCE = 1e3  # the overhead node sets 1e6 when it does not know the heading


def load_merging_map() -> None:
    """Make `merging_map/` importable: MERGING_MAP_PATH if set, else the first parent folder
    of this file that contains it (the repo root, e.g. /ws in the container)."""
    try:
        import fusion_carte  # noqa: F401

        return
    except ImportError:
        pass
    candidates = [os.environ.get('MERGING_MAP_PATH')]
    candidates += [str(p / 'merging_map') for p in Path(__file__).resolve().parents]
    for candidate in candidates:
        if candidate and (Path(candidate) / 'fusion_carte.py').is_file():
            sys.path.insert(0, candidate)
            return
    raise ImportError(
        'merging_map/ not found: set MERGING_MAP_PATH to the folder that holds fusion_carte.py'
    )


def ros_grid_to_fusion(data, width: int, height: int) -> np.ndarray:
    """OccupancyGrid data (row-major, row 0 at the origin = south) -> -1/0/1 matrix.

    Unknown stays -1; the overhead camera's obstacle value (100) becomes 1; its rover (50) and
    margin (99) cells become free. Any other value follows the usual thresholds (>= 65 is an
    obstacle), for maps coming from somewhere else."""
    grid = np.asarray(data, dtype=np.int16).reshape(height, width)
    fusion = np.zeros((height, width), dtype=np.int8)
    fusion[grid < 0] = -1
    fusion[(grid >= 65) & (grid != ROVER) & (grid != MARGIN)] = 1
    return fusion


def tiles_from_ros_grid(data, width, height, resolution, origin_x, origin_y, carte, t=0.0):
    """Drone tile messages for `carte.recevoir()` from an OccupancyGrid."""
    from fusion_carte import decouper

    if not math.isclose(resolution, carte.res_fine, rel_tol=1e-3):
        raise ValueError(
            f'/map has {resolution} m cells; the fusion expects {carte.res_fine} m '
            '(cell_size: 0.05 in overhead.yaml).'
        )
    matrix = ros_grid_to_fusion(data, width, height)
    return decouper(matrix, origin_x, origin_y, carte.zmax, carte, 'drone', t)


def grid_area(width, height, resolution, origin_x, origin_y):
    """(x0, y0, x1, y1) covered by the whole grid, unknown cells included."""
    return (origin_x, origin_y, origin_x + width * resolution, origin_y + height * resolution)


def seen_area(data, width, height, resolution, origin_x, origin_y):
    """(x0, y0, x1, y1) around the cells the camera saw: the planning area. None if it saw
    nothing.

    The overhead camera's /map is larger than its view; the cells around the view are unknown
    (-1). Planning on the whole grid would route the rover through ground nobody has seen,
    where the camera cannot follow it either."""
    known = np.asarray(data, dtype=np.int16).reshape(height, width) >= 0
    if not known.any():
        return None
    rows, cols = np.flatnonzero(known.any(axis=1)), np.flatnonzero(known.any(axis=0))
    return (
        origin_x + float(cols[0]) * resolution,
        origin_y + float(rows[0]) * resolution,
        origin_x + float(cols[-1] + 1) * resolution,
        origin_y + float(rows[-1] + 1) * resolution,
    )


def heading_from_pose(qz: float, qw: float, yaw_variance: float) -> float | None:
    """Heading (rad) of a flat pose, or None when its variance says it is not known."""
    if yaw_variance >= UNKNOWN_HEADING_VARIANCE:
        return None
    return 2.0 * math.atan2(qz, qw)


def fusion_to_ros_grid(values: np.ndarray) -> list[int]:
    """Merged -1/0/1 grid (row 0 = south) -> OccupancyGrid data (row-major)."""
    out = np.where(values == 1, OCCUPIED, np.where(values == 0, 0, UNKNOWN))
    return out.astype(np.int8).ravel().tolist()


def decision_summary(decision: dict) -> dict:
    """What the merged planner decided, without its large arrays, for a JSON message."""
    summary = {'action': decision['action']}
    for key in ('tuiles', 'a_verifier'):
        if decision.get(key):
            summary[key] = [list(t) for t in decision[key]]
    for key in ('point', 'cible', 'point_de_vue'):
        if decision.get(key) is not None:
            summary[key] = [round(float(v), 3) for v in decision[key]]
    if decision.get('chemin'):
        summary['path_length'] = round(
            sum(
                math.dist(a, b)
                for a, b in zip(decision['chemin'], decision['chemin'][1:], strict=False)
            ),
            3,
        )
    return summary
