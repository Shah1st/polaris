"""Download what the simulation needs that is not stored in git.

    python sim/fetch_scene.py

- The RaspRover meshes (STL) from Waveshare's ugv_ws repository (not redistributed here),
  pinned to one commit.
- The apartment scene (furniture meshes, textures, true occupancy maps), ~13 MB,
  from this repository's GitHub release `sim-apartment-v1` (assets: Poly Haven, CC0),
  checked against its SHA-256.

Everything lands under sim/scenes/ and sim/assets/ (both git-ignored). Safe to re-run: what is
already there is kept, and nothing is written until its download is complete.
"""

import hashlib
import io
import os
import shutil
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE_URL = (
    'https://github.com/Shah1st/polaris/releases/download/'
    'sim-apartment-v1/apartment_scene_v1.zip'
)
SCENE_SHA256 = '6ebdfe8f4bb523e56d58c05afde8eaa9380d7c204d5fdbdeabb342c8c87a7081'
# ugv_ws branch ros2-humble-develop as of 2026-11-14.
WAVESHARE = (
    'https://raw.githubusercontent.com/waveshareteam/ugv_ws/'
    'f0b3ad9c16e7acb6d6145f261116d71bf6338f4a/'
    'src/ugv_main/ugv_description/meshes/rasp_rover/'
)
ROVER_MESHES = [
    'base_link',
    'base_lidar_link',
    'pt_base_link',
    'pt_link1',
    'pt_link2',
    'pt_camera_link',
    'left_up_wheel_link',
    'left_down_wheel_link',
    'right_up_wheel_link',
    'right_down_wheel_link',
]


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'polaris-sim'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def write_atomic(path: str, data: bytes) -> None:
    with open(path + '.tmp', 'wb') as f:
        f.write(data)
    os.replace(path + '.tmp', path)


def fetch_meshes() -> None:
    meshes = os.path.join(HERE, 'assets', 'rasp_rover', 'meshes')
    os.makedirs(meshes, exist_ok=True)
    for name in ROVER_MESHES:
        path = os.path.join(meshes, name + '.stl')
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            write_atomic(path, get(WAVESHARE + name + '.stl'))
    print('rover meshes: ok ->', meshes)


def fetch_scene() -> None:
    scene = os.path.join(HERE, 'scenes', 'apartment')
    if os.path.exists(os.path.join(scene, 'mujoco', 'manifest.json')):
        print('apartment scene: already there')
        return
    print('apartment scene: downloading ~13 MB ...')
    data = get(SCENE_URL)
    digest = hashlib.sha256(data).hexdigest()
    if digest != SCENE_SHA256:
        raise RuntimeError(f'apartment scene: unexpected SHA-256 {digest}, not extracted')
    tmp = scene + '.tmp'
    shutil.rmtree(tmp, ignore_errors=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
    shutil.rmtree(scene, ignore_errors=True)
    os.replace(tmp, scene)
    print('apartment scene: ok ->', scene)


def main() -> int:
    failed = False
    for step in (fetch_meshes, fetch_scene):
        try:
            step()
        except Exception as e:  # report every failure, then exit non-zero
            print(f'{step.__name__}: failed: {e}', file=sys.stderr)
            failed = True
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
