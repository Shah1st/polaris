"""Estimates the rover pose in the map frame from the overhead camera.

With `detection: marker` it detects the ArUco marker on top of the rover and maps it onto the
floor plane with the homography from guidance_calib. With `detection: size` the rover has no
marker: global_map_node finds it among the obstacle blobs and publishes the pose, and this node
stays idle.
"""

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from guidance_calib.aruco_compat import DEFAULT_DICTIONARY, detect_markers
from guidance_rover_pose.pose import rover_pose_from_marker

HOMOGRAPHY_TOPIC = '/overhead/image_to_map'
IMAGE_TOPIC = '/overhead/image_raw'
POSE_TOPIC = '/rover/pose_overhead'


class RoverPoseNode(Node):
    """Publishes the rover pose measured by the overhead camera."""

    def __init__(self) -> None:
        super().__init__('rover_pose_node')
        self.declare_parameter('detection', 'size')
        self.declare_parameter('dictionary', DEFAULT_DICTIONARY)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('rover_marker_id', 10)
        self.declare_parameter('yaw_offset', 0.0)
        self.declare_parameter('position_std', 0.03)
        self.declare_parameter('yaw_std', 0.05)

        self._bridge = CvBridge()
        self._homography: np.ndarray | None = None
        if self.get_parameter('detection').value != 'marker':
            self.get_logger().info('rover_pose_node idle: global_map_node finds the rover by size')
            return
        self._publisher = self.create_publisher(PoseWithCovarianceStamped, POSE_TOPIC, 10)
        self.create_subscription(
            Float64MultiArray,
            HOMOGRAPHY_TOPIC,
            self._on_homography,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, qos_profile_sensor_data)
        # TODO(rover_pose): correct the parallax from the marker's height above the floor.
        self.get_logger().info('rover_pose_node waiting for the image and the homography')

    def _on_homography(self, msg: Float64MultiArray) -> None:
        self._homography = np.array(msg.data, dtype=np.float64).reshape(3, 3)

    def _on_image(self, msg: Image) -> None:
        if self._homography is None:
            return
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detections = detect_markers(image, self.get_parameter('dictionary').value)
        corners = detections.get(self.get_parameter('rover_marker_id').value)
        if corners is None:
            self.get_logger().warn('rover marker not visible', throttle_duration_sec=5.0)
            return
        x, y, yaw = rover_pose_from_marker(
            corners, self._homography, self.get_parameter('yaw_offset').value
        )

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = msg.header.stamp
        pose_msg.header.frame_id = self.get_parameter('frame_id').value
        pose_msg.pose.pose.position.x = x
        pose_msg.pose.pose.position.y = y
        pose_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        position_variance = self.get_parameter('position_std').value ** 2
        covariance = [0.0] * 36
        covariance[0] = position_variance
        covariance[7] = position_variance
        covariance[35] = self.get_parameter('yaw_std').value ** 2
        pose_msg.pose.covariance = covariance
        self._publisher.publish(pose_msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RoverPoseNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
