#!/usr/bin/env bash
# Fast test loop: the plain-Python tests of the overhead camera, the map fusion and the
# dashboard's maps, without building the ROS workspace (about half a minute).
#
#   scripts/test_fast.sh                 in the dev container, from the repo root
#   scripts/test_fast.sh -k scale        any pytest option
#   docker run --rm -v "$PWD":/ws polaris-dev scripts/test_fast.sh     from the host
#
# The ROS node tests are left out: they need `colcon build` (see the README).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="src/guidance_calib:src/guidance_global_map:src/guidance_rover_pose:src/guidance_supervisor:merging_map:${PYTHONPATH}"
python3 -m pytest -q -p no:cacheprovider \
    --ignore-glob='*_node.py' \
    src/guidance_calib/test/test_ground_plane.py \
    src/guidance_global_map/test/test_segmentation.py \
    src/guidance_global_map/test/test_scale_check.py \
    src/guidance_global_map/test/test_drone.py \
    src/guidance_global_map/test/test_camera_source.py \
    src/guidance_rover_pose/test/test_pose.py \
    src/guidance_supervisor/test/test_fusion_bridge.py \
    merging_map/test_lancer_overhead.py \
    merging_map/test_fusion_carte.py \
    sim \
    "$@"
