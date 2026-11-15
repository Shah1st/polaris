#!/bin/bash
# Run inside the polaris dev image, workspace mounted at /ws.
# Builds and tests the workspace, then checks that rover_vo_bridge_node really
# publishes /rover/vo/odom and /rover/points when fed by a fake mapper.
set -eo pipefail
source /opt/ros/jazzy/setup.bash
cd /ws

echo "=== rosdep"
rosdep update --rosdistro jazzy > /dev/null
rosdep install --from-paths src --ignore-src -y -r

echo "=== colcon build"
colcon build --symlink-install
source install/setup.bash

echo "=== colcon test"
colcon test --event-handlers console_direct+ || true
colcon test-result --verbose

echo "=== live check of rover_vo_bridge_node against a fake mapper"
python3 /smoke/fake_mapper.py 7400 &
FAKE=$!
ros2 run guidance_rover_vo rover_vo_bridge_node --ros-args -p host:=127.0.0.1 -p port:=7400 &
NODE=$!
sleep 5
echo "--- one /rover/vo/odom message:"
timeout 20 ros2 topic echo --once /rover/vo/odom nav_msgs/msg/Odometry | head -30
echo "--- /rover/points rate and size:"
timeout 20 ros2 topic echo --once /rover/points sensor_msgs/msg/PointCloud2 --no-arr | grep -E "frame_id|width|height|point_step"
echo "--- TF odom -> base_link:"
timeout 20 ros2 run tf2_ros tf2_echo odom base_link 2>&1 | head -6 || true
kill $NODE $FAKE || true
# `ros2 run` and `timeout | head` can leave the real processes behind: clear them all.
pkill -f /ws/install/ || true
pkill -f fake_mapper.py || true
pkill -f tf2_echo || true
sleep 3

echo "=== guidance.launch.py: rover_vo bridge only with rover_vo:=true"
for arg in "" "rover_vo:=true"; do
  # Background jobs ignore SIGINT in a script: stop the launch with TERM, then KILL.
  timeout -s TERM -k 5 15 ros2 launch guidance_bringup guidance.launch.py $arg > /tmp/launch.log 2>&1 &
  L=$!
  sleep 10
  echo "--- nodes with '${arg:-default}':"
  ros2 node list --no-daemon | sort
  wait $L || true
  pkill -f /ws/install/ || true
  sleep 3
done
echo "=== SMOKE TEST DONE"
