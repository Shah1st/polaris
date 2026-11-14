import rclpy

from guidance_rover_pose.rover_pose_node import RoverPoseNode


def test_rover_pose_node_starts() -> None:
    rclpy.init()
    try:
        node = RoverPoseNode()
        assert node.get_name() == 'rover_pose_node'
        node.destroy_node()
    finally:
        rclpy.shutdown()
