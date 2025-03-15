import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='perception',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),
        launch_ros.actions.Node(
            package='perception',
            executable='camera_splitter_node',
            name='camera_splitter_node',
            output='screen'
        ),
        launch_ros.actions.Node(
            package='perception',
            executable='manual_focus_node',
            name='manual_focus_node',
            output='screen'
        ),
    ])