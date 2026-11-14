import rclpy

from guidance_supervisor.supervisor_node import SupervisorNode


def test_supervisor_node_starts() -> None:
    rclpy.init()
    try:
        node = SupervisorNode()
        assert node.get_name() == 'supervisor_node'
        node.destroy_node()
    finally:
        rclpy.shutdown()
