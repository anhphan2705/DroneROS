import launch
import launch_ros.actions

def generate_launch_description():
    camera_rectification_node = launch_ros.actions.Node(
        package='calibration',
        executable='camera_rectification_node',
        name='camera_rectification_node',
        output='screen'
    )

    return launch.LaunchDescription([
        camera_rectification_node,
    ])