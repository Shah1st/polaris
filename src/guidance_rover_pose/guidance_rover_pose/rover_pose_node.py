"""Estimates the rover pose in the map frame from the overhead camera.

Placeholder node: it only starts and logs. No logic yet.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class RoverPoseNode(Node):
    """Placeholder for the rover_pose node."""

    def __init__(self) -> None:
        super().__init__('rover_pose_node')
        # TODO(rover_pose): Detect the rover in the overhead image (e.g. a fiducial marker on its
        #   top).
        # TODO(rover_pose): Convert the pixel pose to the map frame using the guidance_calib
        #   output.
        # TODO(rover_pose): Publish geometry_msgs/PoseWithCovarianceStamped for robot_localization
        #   to fuse with wheel odometry.
        self.get_logger().info('rover_pose_node started (placeholder, no logic yet)')


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
