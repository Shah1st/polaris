"""Mission supervisor: goals, mission state machine and health monitoring.

Placeholder node: it only starts and logs. No logic yet.
"""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class SupervisorNode(Node):
    """Placeholder for the supervisor node."""

    def __init__(self) -> None:
        super().__init__('supervisor_node')
        # TODO(supervisor): Accept a goal (RViz 2D goal, service or topic) and send it to Nav2
        #   (nav2_msgs/action/NavigateToPose).
        # TODO(supervisor): Track the mission state (idle / planning / following / blocked / done)
        #   and handle Nav2 failures and recoveries.
        # TODO(supervisor): Monitor that /map, the rover pose and the depth camera stream are
        #   alive; stop the rover if any goes stale.
        self.get_logger().info('supervisor_node started (placeholder, no logic yet)')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SupervisorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
