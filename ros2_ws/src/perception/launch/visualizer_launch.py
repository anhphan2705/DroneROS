import launch
import launch_ros.actions


def generate_launch_description():
    overlay_0 = launch_ros.actions.Node(
        package='perception',
        executable='bbox_overlay_node',
        name='bbox_overlay_0',
        output='screen',
        parameters=[
            {'image_topic': '/camera/rectified/split_0'},
            {'detection_topic': '/yolo/detections_0'},
            {'output_topic': '/camera/yolo_overlay_0'}
        ]
    )

    overlay_1 = launch_ros.actions.Node(
        package='perception',
        executable='bbox_overlay_node',
        name='bbox_overlay_1',
        output='screen',
        parameters=[
            {'image_topic': '/camera/rectified/split_2'},
            {'detection_topic': '/yolo/detections_1'},
            {'output_topic': '/camera/yolo_overlay_1'}
        ]
    )

    depth_viz_0 = launch_ros.actions.Node(
        package='perception',
        executable='depth_visualizer_node',
        name='depth_visualizer_0',
        output='screen',
        parameters=[
            {'depth_topic': '/camera/depth_map_0'},
            {'output_topic': '/camera/depth_vis_0'},
            {'max_depth': 10.0}
        ]
    )

    depth_viz_1 = launch_ros.actions.Node(
        package='perception',
        executable='depth_visualizer_node',
        name='depth_visualizer_1',
        output='screen',
        parameters=[
            {'depth_topic': '/camera/depth_map_1'},
            {'output_topic': '/camera/depth_vis_1'},
            {'max_depth': 10.0}
        ]
    )
    
    return launch.LaunchDescription([
        overlay_0,
        # overlay_1,
        # depth_viz_0,
        # depth_viz_1,
    ])