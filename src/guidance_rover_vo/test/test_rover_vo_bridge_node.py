import math

import rclpy

from guidance_rover_vo.rover_vo_bridge_node import (
    RoverVoBridgeNode,
    diagonal_covariance,
    yaw_to_quaternion,
)


def test_rover_vo_bridge_node_starts_without_mapper() -> None:
    # No mapper is listening: the node must still start and keep retrying.
    rclpy.init()
    try:
        node = RoverVoBridgeNode()
        assert node.get_name() == 'rover_vo_bridge_node'
        node.destroy_node()
    finally:
        rclpy.shutdown()


def test_yaw_to_quaternion() -> None:
    x, y, z, w = yaw_to_quaternion(math.pi / 2)
    assert (x, y) == (0.0, 0.0)
    assert math.isclose(z, math.sqrt(0.5)) and math.isclose(w, math.sqrt(0.5))


def test_diagonal_covariance() -> None:
    cov = diagonal_covariance((1, 2, 3, 4, 5, 6))
    assert [cov[i * 7] for i in range(6)] == [1, 2, 3, 4, 5, 6]
    assert sum(cov) == 21
