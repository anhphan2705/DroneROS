import launch
from launch import LaunchDescription
import launch_ros.actions

from ament_index_python.packages import get_package_share_directory
import os

calib_pkg_share = get_package_share_directory('perc')

calib_file_0 = os.path.join(
    calib_pkg_share,
    'calibration',
    'calibrated_params',
    'stereo_calibration_params_pair_0_2026-01-18_22-56-19.yml'
)

calib_file_1 = os.path.join(
    calib_pkg_share,
    'calibration',
    'calibrated_params',
    'stereo_calibration_params_pair_1_2026-01-18_22-57-06.yml'
)

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

    perception_gpu_node = launch_ros.actions.Node(
        package='perc',
        executable='perception_gpu_node',
        name='perception_gpu_node',
        output='screen',
        parameters=[
            {'sensor_id': 0},
            {'width': 1920},
            {'height': 1080},
            {'fps': 60},
            {'calibration_file_0': calib_file_0},
            {'calibration_file_1': calib_file_1},
        ]
    )


    # ------------------------------------------------
    # Launch description
    # ------------------------------------------------
    return LaunchDescription([
        # camera_capture_node,
        perception_gpu_node,
    ])
