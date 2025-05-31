from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "namespace",
                default_value="",
                description="Namespace for node"
            ),
            DeclareLaunchArgument(
                "image_transport",
                default_value="raw",
                description="Image transport type (raw, compressed)",
            ),
            DeclareLaunchArgument(
                "model_size",
                default_value="Base",
                description="Model size: Base, Small, or Large",
            ),
            DeclareLaunchArgument(
                "save_to_ply",
                default_value="False",
                description="Save pointcloud to PLY file",
            ),
            DeclareLaunchArgument(
                "downsample",
                default_value="3",
                description="Downsample factor for pointcloud",
            ),
            DeclareLaunchArgument(
                "use_sim_time", default_value="False", description="Use simulation time"
            ),
            Node(
                package="ros2_depthanything",
                executable="depth_to_pointcloud",
                name="depth_to_pointcloud",
                namespace=LaunchConfiguration("namespace"),
                output="screen",
                remappings=[
                    ("image", "image_compressed"),
                    ("camera_info", "camera_info"),
                    ("pointcloud", "pointcloud"),
                ],
                parameters=[
                    {"image_transport": LaunchConfiguration("image_transport")},
                    {"save_to_ply": LaunchConfiguration("save_to_ply")},
                    {"model_size": LaunchConfiguration("model_size")},
                    {"downsample": LaunchConfiguration("downsample")},
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            ),
        ]
    )
