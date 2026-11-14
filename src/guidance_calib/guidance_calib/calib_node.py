"""Overhead camera calibration: intrinsics and image-to-ground-plane mapping.

Placeholder node: it only starts and logs. No logic yet.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class CalibNode(Node):
    """Placeholder for the calib node."""

    def __init__(self) -> None:
        super().__init__('calib_node')
        # TODO(calib): Load overhead camera intrinsics (sensor_msgs/CameraInfo) or run an intrinsic
        #   calibration.
        # TODO(calib): Estimate the image -> ground plane homography (e.g. from fiducial markers at
        #   known positions).
        # TODO(calib): Publish the map -> overhead_camera transform and/or the homography for
        #   guidance_global_map and guidance_rover_pose.
        self.get_logger().info('calib_node started (placeholder, no logic yet)')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CalibNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
