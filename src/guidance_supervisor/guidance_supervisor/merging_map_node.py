"""Map fusion node: wraps `merging_map/` (plain Python) for the ROS side.

Subscribes to the overhead camera's map (/map) and rover pose (/rover/pose_overhead), and to
a goal (/goal_pose, e.g. RViz "2D Goal Pose", or the goal_x / goal_y parameters). About once
a second it runs the fusion's planner and publishes:

- /fusion/map       nav_msgs/OccupancyGrid  the merged map (frame `map`, 5 cm cells)
- /fusion/plan      nav_msgs/Path           the planned path, empty when there is none
- /fusion/decision  std_msgs/String         JSON: action (SUIVRE, DEMANDER_DRONE,
                                            OBJECTIF_DEGRADE...), its details, and the
                                            rover pose it planned from

It plans inside the area the camera saw: the cells of /map that are not unknown.

It plans; it does not drive the rover. Driving stays with `merging_map/rover_reel.py`
(`lancer_overhead.py`) or, later, Nav2 following /fusion/plan.
"""

import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from guidance_supervisor.fusion_bridge import (
    decision_summary,
    fusion_to_ros_grid,
    heading_from_pose,
    load_merging_map,
    seen_area,
    tiles_from_ros_grid,
)

load_merging_map()
from fusion_carte import CarteFusion  # noqa: E402


class MergingMapNode(Node):
    """Feeds the map fusion from ROS topics and publishes its plan."""

    def __init__(self) -> None:
        super().__init__('merging_map_node')
        # Small-area settings (docs/overhead-camera.md). rayon_robot is the circle around the
        # whole 0.20 x 0.23 m rover (hypot / 2 = 0.152 m), not the doc's 0.115 half-length.
        self.declare_parameter('niveau_plan', 4)
        self.declare_parameter('rayon_robot', 0.155)
        # on top of the radius, for what the planner cannot know (path tracking error)
        self.declare_parameter('marge_securite', 0.03)
        self.declare_parameter('tolerance', 0.15)
        self.declare_parameter('goal_x', math.nan)
        self.declare_parameter('goal_y', math.nan)
        self.declare_parameter('period', 1.0)
        p = self.get_parameter
        self.carte = CarteFusion(
            niveau_plan=p('niveau_plan').value,
            rayon_robot=p('rayon_robot').value,
            marge_securite=p('marge_securite').value,
        )
        self.tolerance = p('tolerance').value
        gx, gy = p('goal_x').value, p('goal_y').value
        self.goal = None if math.isnan(gx) or math.isnan(gy) else (gx, gy)
        self.zone = None
        self.position = None
        self.heading = None

        self.create_subscription(OccupancyGrid, '/map', self.on_map, 1)
        self.create_subscription(
            PoseWithCovarianceStamped, '/rover/pose_overhead', self.on_pose, 10
        )
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 1)
        self.map_pub = self.create_publisher(OccupancyGrid, '/fusion/map', 1)
        self.plan_pub = self.create_publisher(Path, '/fusion/plan', 1)
        self.decision_pub = self.create_publisher(String, '/fusion/decision', 1)
        self.create_timer(p('period').value, self.step)
        self.get_logger().info('merging_map_node started')

    # ---- inputs ------------------------------------------------------------------------
    def on_map(self, msg: OccupancyGrid) -> None:
        info = msg.info
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        try:
            tiles = tiles_from_ros_grid(
                msg.data,
                info.width,
                info.height,
                info.resolution,
                info.origin.position.x,
                info.origin.position.y,
                self.carte,
                t,
            )
        except ValueError as e:
            self.get_logger().error(str(e))
            return
        for tile in tiles:
            self.carte.recevoir(tile)
        # plan only where the camera saw the floor, not on the unknown cells around its view
        self.zone = seen_area(
            msg.data,
            info.width,
            info.height,
            info.resolution,
            info.origin.position.x,
            info.origin.position.y,
        )

    def on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        pose = msg.pose.pose
        self.position = (pose.position.x, pose.position.y)
        self.heading = heading_from_pose(
            pose.orientation.z, pose.orientation.w, msg.pose.covariance[35]
        )

    def on_goal(self, msg: PoseStamped) -> None:
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.get_logger().info(f'goal set to {self.goal}')

    # ---- planning ----------------------------------------------------------------------
    def step(self) -> None:
        if self.zone is None or self.position is None or self.goal is None:
            return
        decision = self.carte.decider(
            self.position, self.goal, zone=self.zone, tolerance=self.tolerance
        )
        stamp = self.get_clock().now().to_msg()
        self.publish_map(stamp)
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = stamp
        for x, y in decision.get('chemin') or []:
            point = PoseStamped()
            point.header = path.header
            point.pose.position.x, point.pose.position.y = float(x), float(y)
            point.pose.orientation.w = 1.0
            path.poses.append(point)
        self.plan_pub.publish(path)
        summary = decision_summary(decision)
        # where the planner started from; the heading is null when the camera does not know it
        summary['rover'] = {
            'x': round(self.position[0], 3),
            'y': round(self.position[1], 3),
            'heading': None if self.heading is None else round(self.heading, 3),
        }
        self.decision_pub.publish(String(data=json.dumps(summary)))

    def publish_map(self, stamp) -> None:
        x0, y0, x1, y1 = self.zone
        grille = self.carte.grille(x0, y0, x1, y1)
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = stamp
        msg.info.resolution = float(self.carte.res_fine)
        msg.info.height, msg.info.width = grille.val.shape
        msg.info.origin.position.x = grille.j0 * grille.res
        msg.info.origin.position.y = grille.i0 * grille.res
        msg.info.origin.orientation.w = 1.0
        msg.data = fusion_to_ros_grid(grille.val)
        self.map_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MergingMapNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
