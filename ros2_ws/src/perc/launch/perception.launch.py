import launch
from launch import LaunchDescription
import launch_ros.actions

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    perc_share = get_package_share_directory('perc')
    calib_dir = os.path.join(perc_share, 'calibration')

    camera_capture_node = launch_ros.actions.Node(
        package='perc',
        executable='camera_capture_node',
        name='camera_capture_node',
        output='screen',
        parameters=[
            # Camera selection
            {'sensor_id': 0},

            # Image properties
            {'width': 1920},
            {'height': 1080},
            {'fps': 60},

            # ROS metadata
            {'frame_id': 'camera'},
            {'topic': '/camera/image_nv12'},

            # Debug / visualization
            {'debug_rgb': True},
        ]
    )

    # ------------------------------------------------
    # Launch description
    # ------------------------------------------------
    return LaunchDescription([
        camera_capture_node,
    ])
