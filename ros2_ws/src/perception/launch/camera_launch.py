import launch
import launch_ros.actions

def generate_launch_description():
    camera_node = launch_ros.actions.Node(
        package='perception',
        executable='camera_node',
        name='camera_node',
        output='screen'
    )

    camera_splitter_node = launch_ros.actions.Node(
        package='perception',
        executable='camera_splitter_node',
        name='camera_splitter_node',
        output='screen'
    )

    manual_focus_node = launch_ros.actions.Node(
        package='perception',
        executable='manual_focus_node',
        name='manual_focus_node',
        output='screen'
    )

    sync_capture_node = launch_ros.actions.Node(
        package='perception',
        executable='sync_capture_node',
        name='sync_capture_node',
        output='screen'
    )

    delayed_manual_focus_node = launch.actions.TimerAction(
        period=10.0,
        actions=[manual_focus_node]
    )

    return launch.LaunchDescription([
        camera_node,
        camera_splitter_node,
        delayed_manual_focus_node,
        sync_capture_node,
    ])