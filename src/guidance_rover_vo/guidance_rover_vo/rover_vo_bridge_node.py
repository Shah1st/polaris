"""Bridge from the rover mapper (monocular 3D mapping) into ROS.

The mapper runs natively on the host (it needs the host GPU) and streams the
rover state over TCP (see `protocol.py`). This node connects to it and
publishes:

* `/rover/vo/odom` (nav_msgs/Odometry): base_link pose in the `odom` frame,
  for robot_localization to fuse with the overhead-camera pose. Not published
  while the mapper is lost.
* `/rover/points` (sensor_msgs/PointCloud2, base_link): 3D points in front of
  the rover with people and other moving objects removed, for the Nav2 local
  costmap (filter the floor with min_obstacle_height there).
* TF odom -> base_link, if `publish_tf` is true (default, until the EKF owns
  that transform).

Caveats: the metric scale comes from the mapper's floor calibration (about
+-15% on the one sequence measured); the pose may jump once or twice in the
first seconds when the mapper calibrates (logged).
"""

from __future__ import annotations

import math
import socket
import threading

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster

from guidance_rover_vo.protocol import RoverState, StreamReader

# Planar rover: x, y, yaw are measured; z, roll, pitch are fixed (large variance
# so the EKF ignores them if they are not already disabled in its config).
_POSE_VAR = (0.01, 0.01, 1e3, 1e3, 1e3, 0.0025)
_TWIST_VAR = (0.02, 0.02, 1e3, 1e3, 1e3, 0.01)


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion for a rotation of `yaw` about z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def diagonal_covariance(variances: tuple[float, ...]) -> list[float]:
    cov = [0.0] * 36
    for i, v in enumerate(variances):
        cov[i * 7] = v
    return cov


class RoverVoBridgeNode(Node):
    """Publishes the rover mapper's pose and local points."""

    def __init__(self) -> None:
        super().__init__('rover_vo_bridge_node')
        self.declare_parameter('host', 'host.docker.internal')
        self.declare_parameter('port', 7400)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_tf', True)
        self._host = self.get_parameter('host').value
        self._port = int(self.get_parameter('port').value)
        self._odom_frame = self.get_parameter('odom_frame').value
        self._base_frame = self.get_parameter('base_frame').value

        self._odom_pub = self.create_publisher(Odometry, '/rover/vo/odom', 10)
        self._points_pub = self.create_publisher(
            PointCloud2, '/rover/points', qos_profile_sensor_data
        )
        self._tf = TransformBroadcaster(self) if self.get_parameter('publish_tf').value else None

        self._lock = threading.Lock()
        self._latest: RoverState | None = None
        self._prev: RoverState | None = None
        self._map_version: int | None = None
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.create_timer(1.0 / 30.0, self._publish_latest)
        self.get_logger().info(f'waiting for the rover mapper on {self._host}:{self._port}')

    # -- TCP side (background thread) ----------------------------------------
    def _read_loop(self) -> None:
        while self._running:
            try:
                with socket.create_connection((self._host, self._port), timeout=2.0) as sock:
                    sock.settimeout(5.0)
                    self.get_logger().info('connected to the rover mapper')
                    reader = StreamReader(sock)
                    while self._running:
                        state = reader.read()
                        with self._lock:
                            self._latest = state
            except (OSError, ConnectionError, ValueError) as e:
                if self._running:
                    self.get_logger().warning(
                        f'rover mapper not reachable ({e}), retrying', throttle_duration_sec=10.0
                    )
                    threading.Event().wait(1.0)

    # -- ROS side (executor thread) ------------------------------------------
    def _publish_latest(self) -> None:
        with self._lock:
            state, self._latest = self._latest, None
        if state is None:
            return
        if self._map_version is not None and state.map_version != self._map_version:
            self.get_logger().info('rover mapper re-expressed its frame: odom may jump once')
            self._prev = None
        self._map_version = state.map_version
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self._odom_frame)

        if state.lost:
            self.get_logger().warning(
                'rover mapper lost, not publishing odometry', throttle_duration_sec=5.0
            )
            self._prev = None
            return

        odom = Odometry(header=header, child_frame_id=self._base_frame)
        odom.pose.pose.position.x = state.x
        odom.pose.pose.position.y = state.y
        q = yaw_to_quaternion(state.yaw)
        o = odom.pose.pose.orientation
        o.x, o.y, o.z, o.w = q
        odom.pose.covariance = diagonal_covariance(_POSE_VAR)
        if self._prev is not None and state.stamp > self._prev.stamp:
            dt = state.stamp - self._prev.stamp
            dx, dy = state.x - self._prev.x, state.y - self._prev.y
            c, s = math.cos(state.yaw), math.sin(state.yaw)
            odom.twist.twist.linear.x = (c * dx + s * dy) / dt  # in base_link
            odom.twist.twist.linear.y = (-s * dx + c * dy) / dt
            dyaw = math.atan2(
                math.sin(state.yaw - self._prev.yaw), math.cos(state.yaw - self._prev.yaw)
            )
            odom.twist.twist.angular.z = dyaw / dt
            odom.twist.covariance = diagonal_covariance(_TWIST_VAR)
        self._prev = state
        self._odom_pub.publish(odom)

        if self._tf is not None:
            t = TransformStamped(header=header, child_frame_id=self._base_frame)
            t.transform.translation.x = state.x
            t.transform.translation.y = state.y
            r = t.transform.rotation
            r.x, r.y, r.z, r.w = q
            self._tf.sendTransform(t)

        if len(state.points):
            cloud_header = Header(stamp=header.stamp, frame_id=self._base_frame)
            self._points_pub.publish(
                point_cloud2.create_cloud_xyz32(cloud_header, state.points.tolist())
            )

    def destroy_node(self) -> None:
        self._running = False
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RoverVoBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
