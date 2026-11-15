import rclpy

from guidance_supervisor.merging_map_node import MergingMapNode


def test_merging_map_node_starts() -> None:
    rclpy.init()
    try:
        node = MergingMapNode()
        assert node.get_name() == 'merging_map_node'
        assert node.carte.niveau_plan == 4
        node.destroy_node()
    finally:
        rclpy.shutdown()
