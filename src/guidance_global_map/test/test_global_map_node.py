import rclpy

from guidance_global_map.global_map_node import GlobalMapNode


def test_global_map_node_starts() -> None:
    rclpy.init()
    try:
        node = GlobalMapNode()
        assert node.get_name() == 'global_map_node'
        node.destroy_node()
    finally:
        rclpy.shutdown()
