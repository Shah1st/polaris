"""Start the whole guidance stack on one machine.

The overhead-camera nodes (calib, global_map, rover_pose) take their parameters from
config/overhead.yaml. Pass overhead_source:=<image or video file> to feed them from a file.
Pass rover_vo:=true to also start the bridge from the rover mapper (guidance_rover_vo).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# (package, executable) for every guidance node.
GUIDANCE_NODES = [
    ('guidance_calib', 'calib_node'),
    ('guidance_global_map', 'global_map_node'),
    ('guidance_rover_pose', 'rover_pose_node'),
    ('guidance_rover_vo', 'rover_vo_bridge_node'),
    ('guidance_supervisor', 'supervisor_node'),
]
# Nodes started only when their launch argument is true (off by default, so a launch that
# does not use them is unchanged).
OPTIONAL_NODES = {'rover_vo_bridge_node': 'rover_vo'}


def generate_launch_description() -> LaunchDescription:
    log_level = LaunchConfiguration('log_level')
    overhead_source = LaunchConfiguration('overhead_source')
    overhead_params = PathJoinSubstitution(
        [FindPackageShare('guidance_bringup'), 'config', 'overhead.yaml']
    )

    nodes = [
        Node(
            package=package,
            executable=executable,
            name=executable,
            output='screen',
            emulate_tty=True,
            parameters=[overhead_params],
            arguments=['--ros-args', '--log-level', log_level],
            condition=(
                IfCondition(LaunchConfiguration(OPTIONAL_NODES[executable]))
                if executable in OPTIONAL_NODES
                else None
            ),
        )
        for package, executable in GUIDANCE_NODES
    ]

    # Plays an image or video file as the overhead camera. Not started when no file is given.
    overhead_file_source = GroupAction(
        condition=IfCondition(PythonExpression(["'", overhead_source, "' != ''"])),
        actions=[
            Node(
                package='image_publisher',
                executable='image_publisher_node',
                name='overhead_camera',
                output='screen',
                parameters=[
                    {
                        'filename': overhead_source,
                        'frame_id': 'overhead_camera',
                        'publish_rate': 5.0,
                    }
                ],
                remappings=[
                    ('image_raw', '/overhead/image_raw'),
                    ('camera_info', '/overhead/camera_info'),
                ],
            )
        ],
    )

    # TODO(bringup): live overhead camera driver (phone stream), publishing /overhead/image_raw.
    # TODO(bringup): rover depth camera driver (feeds the Nav2 local costmap for obstacle
    #   avoidance).
    # TODO(bringup): static TFs (base_link -> depth camera, map -> overhead camera if not
    #   published by guidance_calib).
    # TODO(bringup): robot_localization EKF fusing /rover/pose_overhead with odometry
    #   (config/ekf.yaml).
    # TODO(bringup): Nav2 (planner on /map from guidance_global_map, controller + local
    #   costmap from the depth camera), with config/nav2_params.yaml.

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'log_level', default_value='info', description='Log level for all guidance nodes'
            ),
            DeclareLaunchArgument(
                'overhead_source',
                default_value='',
                description='Image or video file to play as the overhead camera (empty: none)',
            ),
            DeclareLaunchArgument(
                'rover_vo',
                default_value='false',
                description='Start the bridge from the rover mapper (runs on the host)',
            ),
            *nodes,
            overhead_file_source,
        ]
    )
