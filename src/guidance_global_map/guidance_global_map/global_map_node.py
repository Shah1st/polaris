"""Builds the 2D global occupancy map from the overhead camera.

Placeholder node: it only starts and logs. No logic yet.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class GlobalMapNode(Node):
    """Placeholder for the global_map node."""

    def __init__(self) -> None:
        super().__init__('global_map_node')
        # TODO(global_map): Subscribe to the overhead camera image (sensor_msgs/Image) and convert
        #   it with cv_bridge.
        # TODO(global_map): Segment obstacles / free space and warp them onto the ground plane
        #   using the guidance_calib output.
        # TODO(global_map): Publish nav_msgs/OccupancyGrid on /map for the Nav2 global costmap and
        #   planner.
        self.get_logger().info('global_map_node started (placeholder, no logic yet)')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GlobalMapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
