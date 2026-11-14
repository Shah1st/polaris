import rclpy

from guidance_calib.calib_node import CalibNode


def test_calib_node_starts() -> None:
    rclpy.init()
    try:
        node = CalibNode()
        assert node.get_name() == 'calib_node'
        node.destroy_node()
    finally:
        rclpy.shutdown()
