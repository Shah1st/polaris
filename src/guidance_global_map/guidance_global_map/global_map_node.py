"""Builds the 2D global occupancy map from the overhead camera.

Warps the overhead image onto the floor plane with the homography from guidance_calib, segments
obstacles by colour, and publishes a nav_msgs/OccupancyGrid on /map.

With `rover_detection: size` the rover carries no marker: the obstacle blob whose size matches
the rover is taken to be the rover and its position is published on /rover/pose_overhead. Its
heading is not known in this mode.

Map cell values: 0 free, 100 obstacle, 50 the rover itself, 99 margin around obstacles (off by
default), -1 not seen by the camera. The rover's value is not a standard occupancy value: a
planner must treat those cells as its own start, not as a cost.
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from guidance_calib.aruco_compat import DEFAULT_DICTIONARY, detect_markers
from guidance_calib.ground_plane import (
    MapBounds,
    map_to_top_down,
    pixels_to_map,
    warp_to_top_down,
)
from guidance_global_map.pipeline import frame_problem
from guidance_global_map.segmentation import (
    build_cell_grid,
    disc_mask,
    find_blobs,
    find_rover_blob,
    polygons_mask,
    scale_polygon,
    segment_obstacles,
)
from guidance_rover_pose.pose import correct_parallax

HOMOGRAPHY_TOPIC = '/overhead/image_to_map'
IMAGE_TOPIC = '/overhead/image_raw'
POSE_TOPIC = '/rover/pose_overhead'
# Yaw variance published when the heading is unknown, so a filter ignores it.
UNKNOWN_YAW_VARIANCE = 1.0e6
# Printed markers have a white quiet zone around the black square; mask that too.
MARKER_MASK_SCALE = 1.8


def latched_qos() -> QoSProfile:
    return QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class GlobalMapNode(Node):
    """Publishes the global occupancy grid seen by the overhead camera."""

    def __init__(self) -> None:
        super().__init__('global_map_node')
        self.declare_parameter('dictionary', DEFAULT_DICTIONARY)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('x_min', 0.0)
        self.declare_parameter('y_min', 0.0)
        self.declare_parameter('x_max', 2.0)
        self.declare_parameter('y_max', 2.0)
        self.declare_parameter('resolution', 0.005)
        self.declare_parameter('color_threshold', 25.0)
        self.declare_parameter('lightness_weight', 0.7)
        self.declare_parameter('floor_window', 1.0)
        self.declare_parameter('min_obstacle_area', 0.0005)
        self.declare_parameter('cell_size', 0.05)
        self.declare_parameter('margin', 0.0)
        self.declare_parameter('rover_detection', 'size')
        self.declare_parameter('rover_size', [0.20, 0.23])
        self.declare_parameter('rover_size_tolerance', 0.2)
        self.declare_parameter('position_std', 0.03)
        self.declare_parameter('rover_marker_id', 10)
        self.declare_parameter('rover_radius', 0.20)
        # Marker mode: the marker's height above the floor and the camera's, for the parallax.
        self.declare_parameter('marker_height', 0.0)
        self.declare_parameter('camera_height', 1.8)
        self.declare_parameter('update_period', 1.0)

        self._bounds = MapBounds(
            self.get_parameter('x_min').value,
            self.get_parameter('y_min').value,
            self.get_parameter('x_max').value,
            self.get_parameter('y_max').value,
        )
        self._resolution = self.get_parameter('resolution').value
        self._rover_detection = self.get_parameter('rover_detection').value
        if self._rover_detection not in ('size', 'marker'):
            raise ValueError(
                f"rover_detection must be 'size' or 'marker', got {self._rover_detection!r}"
            )

        self._bridge = CvBridge()
        self._latest_image: Image | None = None
        self._homography: np.ndarray | None = None
        self._map_publisher = self.create_publisher(OccupancyGrid, '/map', latched_qos())
        self._top_down_publisher = self.create_publisher(Image, '/overhead/debug/top_down', 1)
        self._mask_publisher = self.create_publisher(Image, '/overhead/debug/mask', 1)
        self._pose_publisher = self.create_publisher(PoseWithCovarianceStamped, POSE_TOPIC, 10)
        self.create_subscription(
            Float64MultiArray, HOMOGRAPHY_TOPIC, self._on_homography, latched_qos()
        )
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, qos_profile_sensor_data)
        self.create_timer(self.get_parameter('update_period').value, self._update)
        # TODO(global_map): temporal fusion over frames, learned segmentation, graded costs.
        # TODO(global_map): rover heading in size mode, from its motion between frames.
        self.get_logger().info('global_map_node waiting for the image and the homography')

    def _on_homography(self, msg: Float64MultiArray) -> None:
        self._homography = np.array(msg.data, dtype=np.float64).reshape(3, 3)

    def _on_image(self, msg: Image) -> None:
        self._latest_image = msg

    def _marker_masks(
        self, image: np.ndarray, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Marker setup: top-down masks of the floor markers and of the rover (None if unseen)."""
        rover_id = self.get_parameter('rover_marker_id').value
        detections = detect_markers(image, self.get_parameter('dictionary').value)
        rover = None
        polygons = []
        for marker_id, corners in detections.items():
            in_map = pixels_to_map(self._homography, corners)
            in_top_down = map_to_top_down(in_map, self._bounds, self._resolution)
            if marker_id == rover_id:
                # The rover spans from its floor point (parallax corrected) to where its top
                # appears: mask around both.
                rows, cols = image.shape[:2]
                nadir = pixels_to_map(self._homography, np.array([[cols / 2.0, rows / 2.0]]))[0]
                floor = correct_parallax(
                    in_map.mean(axis=0, keepdims=True),
                    nadir,
                    self.get_parameter('camera_height').value,
                    self.get_parameter('marker_height').value,
                )
                floor_px = map_to_top_down(floor, self._bounds, self._resolution)[0]
                radius = self.get_parameter('rover_radius').value / self._resolution
                rover = disc_mask(shape, in_top_down.mean(axis=0), radius) | disc_mask(
                    shape, floor_px, radius
                )
            else:
                polygons.append(scale_polygon(in_top_down, MARKER_MASK_SCALE))
        return polygons_mask(shape, polygons), rover

    def _find_rover_by_size(self, obstacles: np.ndarray, image_msg: Image) -> np.ndarray | None:
        """Find the rover among the obstacle blobs and publish its position.

        Returns the rover's mask, or None when no blob has its size.
        """
        rover = find_rover_blob(
            find_blobs(obstacles, self._resolution),
            tuple(self.get_parameter('rover_size').value),
            self.get_parameter('rover_size_tolerance').value,
        )
        if rover is None:
            self.get_logger().warn('no blob of the rover size in view', throttle_duration_sec=5.0)
            return None

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = image_msg.header.stamp
        pose_msg.header.frame_id = self.get_parameter('frame_id').value
        pose_msg.pose.pose.position.x = self._bounds.x_min + rover.center[0] * self._resolution
        pose_msg.pose.pose.position.y = self._bounds.y_max - rover.center[1] * self._resolution
        pose_msg.pose.pose.orientation.w = 1.0
        position_variance = self.get_parameter('position_std').value ** 2
        covariance = [0.0] * 36
        covariance[0] = position_variance
        covariance[7] = position_variance
        covariance[35] = UNKNOWN_YAW_VARIANCE
        pose_msg.pose.covariance = covariance
        self._pose_publisher.publish(pose_msg)
        return rover.mask

    def _update(self) -> None:
        if self._latest_image is None or self._homography is None:
            return
        msg = self._latest_image
        self._latest_image = None
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        problem = frame_problem(image)
        if problem is not None:
            # A black, white or uniform frame would publish "everything free".
            self.get_logger().warn(f'frame rejected: {problem}', throttle_duration_sec=5.0)
            return

        top_down, valid = warp_to_top_down(image, self._homography, self._bounds, self._resolution)
        ignore, rover = None, None
        if self._rover_detection == 'marker':
            ignore, rover = self._marker_masks(image, valid.shape)
            if rover is not None:
                ignore = ignore | rover
        min_area_px = self.get_parameter('min_obstacle_area').value / self._resolution**2
        obstacles = segment_obstacles(
            top_down,
            valid,
            ignore,
            color_threshold=self.get_parameter('color_threshold').value,
            lightness_weight=self.get_parameter('lightness_weight').value,
            min_area_px=int(round(min_area_px)),
            floor_window_px=int(
                round(self.get_parameter('floor_window').value / self._resolution)
            ),
        )
        if self._rover_detection == 'size':
            rover = self._find_rover_by_size(obstacles, msg)
            if rover is not None:
                obstacles = obstacles & ~rover
        cell_size = self.get_parameter('cell_size').value
        margin = self.get_parameter('margin').value
        grid = build_cell_grid(obstacles, valid, self._resolution, cell_size, margin, rover)

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = msg.header.stamp
        grid_msg.header.frame_id = self.get_parameter('frame_id').value
        grid_msg.info.map_load_time = msg.header.stamp
        grid_msg.info.resolution = cell_size
        grid_msg.info.height, grid_msg.info.width = grid.shape
        grid_msg.info.origin.position.x = self._bounds.x_min
        grid_msg.info.origin.position.y = self._bounds.y_min
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = grid.ravel().tolist()
        self._map_publisher.publish(grid_msg)

        self._top_down_publisher.publish(self._bridge.cv2_to_imgmsg(top_down, encoding='bgr8'))
        mask_image = obstacles.astype(np.uint8) * 255
        self._mask_publisher.publish(self._bridge.cv2_to_imgmsg(mask_image, encoding='mono8'))


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
