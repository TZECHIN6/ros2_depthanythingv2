from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Namespace for topics (without leading or trailing /)'
        ),
        DeclareLaunchArgument(
            'image_raw_topic',
            default_value='image_raw',
            description='Raw image topic name'
        ),
        DeclareLaunchArgument(
            'image_compressed_topic',
            default_value='image_compressed',
            description='Compressed image topic name'
        ),
        DeclareLaunchArgument(
            'use_compressed',
            default_value='True',
            description='Use compressed image topic'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='camera_info',
            description='Camera info topic name'
        ),
        DeclareLaunchArgument(
            'pointcloud_topic',
            default_value='pointcloud',
            description='Publish Pointcloud topic name'
        ),
        DeclareLaunchArgument(
            'model_size',
            default_value='Base',
            description='Model size: Base, Small, or Large'
        ),
        DeclareLaunchArgument(
            'save_to_ply',
            default_value='False',
            description='Save pointcloud to PLY file'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='False',
            description='Use simulation time'
        ),
        Node(
            package='ros2_depthanything',
            executable='depth_to_pointcloud',
            name='depth_to_pointcloud',
            output='screen',
            parameters=[
                {'namespace': LaunchConfiguration('namespace')},
                {'image_raw_topic': LaunchConfiguration('image_raw_topic')},
                {'image_compressed_topic': LaunchConfiguration('image_compressed_topic')},
                {'use_compressed': LaunchConfiguration('use_compressed')},
                {'camera_info_topic': LaunchConfiguration('camera_info_topic')},
                {'pointcloud_topic': LaunchConfiguration('pointcloud_topic')},
                {'save_to_ply': LaunchConfiguration('save_to_ply')},
                {'model_size': LaunchConfiguration('model_size')},
                {'use_sim_time': LaunchConfiguration('use_sim_time')}
            ]
        )
    ])