"""Start the whole guidance stack on one machine.

Today this only starts the placeholder nodes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# (package, executable) for every guidance node.
GUIDANCE_NODES = [
    ('guidance_calib', 'calib_node'),
    ('guidance_global_map', 'global_map_node'),
    ('guidance_rover_pose', 'rover_pose_node'),
    ('guidance_rover_vo', 'rover_vo_bridge_node'),
    ('guidance_supervisor', 'supervisor_node'),
]


def generate_launch_description() -> LaunchDescription:
    log_level = LaunchConfiguration('log_level')

    nodes = [
        Node(
            package=package,
            executable=executable,
            name=executable,
            output='screen',
            emulate_tty=True,
            arguments=['--ros-args', '--log-level', log_level],
            # TODO(bringup): pass a params file from config/ once nodes declare parameters.
        )
        for package, executable in GUIDANCE_NODES
    ]

    # TODO(bringup): overhead camera driver (publishes the image + camera_info for calib,
    #   global_map and rover_pose).
    # TODO(bringup): rover depth camera driver (feeds the Nav2 local costmap for obstacle
    #   avoidance).
    # TODO(bringup): static TFs (base_link -> depth camera, map -> overhead camera if not
    #   published by guidance_calib).
    # TODO(bringup): robot_localization EKF fusing the overhead-camera rover pose with
    #   odometry (config/ekf.yaml).
    # TODO(bringup): Nav2 (planner on /map from guidance_global_map, controller + local
    #   costmap from the depth camera), with config/nav2_params.yaml.

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'log_level', default_value='info', description='Log level for all guidance nodes'
            ),
            *nodes,
        ]
    )
