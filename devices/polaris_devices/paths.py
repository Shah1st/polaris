"""Make the repository's plain-Python modules importable (merging_map, the overhead camera).

merging_map is imported as top-level modules (`from fusion_carte import ...`), and the ROS
packages under src/ are importable without ROS when their folder is on sys.path.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FOLDERS = (
    REPO / 'merging_map',
    REPO / 'src' / 'guidance_global_map',
    REPO / 'src' / 'guidance_calib',
    REPO / 'src' / 'guidance_rover_pose',
    REPO / 'sim',
)


def ensure_repo_paths() -> None:
    """Append (not prepend) the folders, so an installed or already-set copy keeps priority."""
    for folder in FOLDERS:
        if folder.is_dir() and str(folder) not in sys.path:
            sys.path.append(str(folder))
