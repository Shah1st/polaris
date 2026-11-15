"""Overhead camera calibration: the image-to-ground-plane homography.

Publishes the 3x3 image -> map homography (row-major, 9 values) on a latched topic. Two modes:

- `camera_height`: no markers. The camera looks straight down from a known height, so the
  scale follows from the height and the lens. The map origin is the image's bottom-left corner.
- `markers`: ArUco markers taped to the floor at known map positions. Handles a tilted camera,
  and serves as ground truth for checking the other mode.
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from guidance_calib.aruco_compat import DEFAULT_DICTIONARY, detect_markers
from guidance_calib.ground_plane import (
    estimate_homography,
    metres_per_pixel_for_view,
    straight_down_homography,
)

HOMOGRAPHY_TOPIC = '/overhead/image_to_map'
IMAGE_TOPIC = '/overhead/image_raw'


def homography_qos() -> QoSProfile:
    """Latched: late subscribers get the last homography."""
    return QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class CalibNode(Node):
    """Publishes the image -> map homography of the overhead camera."""

    def __init__(self) -> None:
        super().__init__('calib_node')
        self.declare_parameter('mode', 'camera_height')
        # Measured ground size of one image pixel; 0 derives it from the height and the lens.
        self.declare_parameter('metres_per_pixel', 0.0)
        self.declare_parameter('camera_height', 1.8)
        self.declare_parameter('focal_length_35mm', 26.0)
        self.declare_parameter('sensor_width_px', 4032)
        self.declare_parameter('sensor_height_px', 3024)
        self.declare_parameter('dictionary', DEFAULT_DICTIONARY)
        self.declare_parameter('marker_ids', [0, 1, 2, 3])
        self.declare_parameter('marker_x', [0.0, 2.0, 2.0, 0.0])
        self.declare_parameter('marker_y', [0.0, 0.0, 1.5, 1.5])
        self.declare_parameter('update_period', 1.0)

        self._mode = self.get_parameter('mode').value
        if self._mode not in ('camera_height', 'markers'):
            raise ValueError(f"mode must be 'camera_height' or 'markers', got {self._mode!r}")
        self._dictionary = self.get_parameter('dictionary').value
        ids = self.get_parameter('marker_ids').value
        xs = self.get_parameter('marker_x').value
        ys = self.get_parameter('marker_y').value
        if not (len(ids) == len(xs) == len(ys)) or len(ids) < 4:
            raise ValueError('marker_ids, marker_x and marker_y need the same length, at least 4')
        self._marker_positions = {
            int(i): (float(x), float(y)) for i, x, y in zip(ids, xs, ys, strict=True)
        }

        self._bridge = CvBridge()
        self._latest_image: Image | None = None
        self._calibrated = False
        self._warned_aspect = False
        self._publisher = self.create_publisher(
            Float64MultiArray, HOMOGRAPHY_TOPIC, homography_qos()
        )
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, qos_profile_sensor_data)
        self.create_timer(self.get_parameter('update_period').value, self._update)
        # TODO(calib): intrinsic calibration and undistortion (ChArUco) for wide lenses.
        # TODO(calib): stamped custom message in guidance_interfaces once the camera moves.
        self.get_logger().info(f'calib_node started in {self._mode} mode')

    def _on_image(self, msg: Image) -> None:
        self._latest_image = msg

    def _metres_per_pixel(self, width: int, height: int) -> float:
        """The measured scale, or the one from the height and lens for this image width."""
        measured = float(self.get_parameter('metres_per_pixel').value)
        if measured > 0.0:
            return measured
        sensor = (
            int(self.get_parameter('sensor_width_px').value),
            int(self.get_parameter('sensor_height_px').value),
        )
        scale, relation = metres_per_pixel_for_view(
            float(self.get_parameter('camera_height').value),
            float(self.get_parameter('focal_length_35mm').value),
            sensor,
            (width, height),
        )
        if relation == 'cropped' and not self._warned_aspect:
            self._warned_aspect = True
            self.get_logger().warn(
                f'image {width}x{height} is not shaped like the sensor {sensor[0]}x{sensor[1]}: '
                "taken as a crop, at the sensor's scale. For a video frame set metres_per_pixel "
                '(ruler on the floor)'
            )
        return scale

    def _update(self) -> None:
        if self._latest_image is None:
            return
        image = self._bridge.imgmsg_to_cv2(self._latest_image, desired_encoding='bgr8')
        self._latest_image = None
        if self._mode == 'camera_height':
            metres_per_pixel = self._metres_per_pixel(image.shape[1], image.shape[0])
            homography = straight_down_homography(metres_per_pixel, image.shape[0])
            detections = {}
        else:
            detections = detect_markers(image, self._dictionary)
            homography = estimate_homography(detections, self._marker_positions)
        if homography is None:
            seen = sorted(set(detections) & set(self._marker_positions))
            self.get_logger().warn(
                f'need 4 floor markers, see {seen}; keeping the last homography',
                throttle_duration_sec=5.0,
            )
            return
        self._publisher.publish(Float64MultiArray(data=np.ravel(homography).tolist()))
        if not self._calibrated:
            self._calibrated = True
            self.get_logger().info('image -> map homography estimated')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CalibNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
