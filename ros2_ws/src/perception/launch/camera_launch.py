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

    camera_rectification_node = launch_ros.actions.Node(
        package='calibration',
        executable='camera_rectification_node',
        name='camera_rectification_node',
        output='screen',
    )

    stereo_depth_node = launch_ros.actions.Node(
        package='perception',
        executable='stereo_depth_node',
        name='stereo_depth_node',
        output='screen'
    )

    yolov8_detection_0 = launch_ros.actions.Node(
        package='perception',
        executable='object_detection_node',
        name='yolo_detection_0',
        output='screen',
        parameters=[
            {'model_path': 'yolov8m_200e.pt'},
            {'image_topic': '/camera/rectified/split_0'},
            {'detection_topic': '/yolo/detections_0'}
        ]
    )

    yolov8_detection_1 = launch_ros.actions.Node(
        package='perception',
        executable='object_detection_node',
        name='yolo_detection_1',
        output='screen',
        parameters=[
            {'model_path': 'best.pt'},
            {'image_topic': '/camera/rectified/split_2'},
            {'detection_topic': '/yolo/detections_1'}
        ]
    )
    
    object_depth_fusion_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='object_depth_fusion_node',
        name='object_depth_fusion_node_0',
        output='screen',
        parameters=[
            {'detection_topic': '/yolo/detections_0'},
            {'depth_topic': '/camera/depth_map_0'},
            {'output_topic': '/yolo/detections_0/depth'}
        ]
    )
    
    object_depth_fusion_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='object_depth_fusion_node',
        name='object_depth_fusion_node',
        output='screen',
        parameters=[
            {'detection_topic': '/yolo/detections_1'},
            {'depth_topic': '/camera/depth_map_1'},
            {'output_topic': '/yolo/detections_1/depth'}
        ]
    )
    
    overlay_0 = launch_ros.actions.Node(
        package='perception',
        executable='bbox_overlay_node',
        name='bbox_overlay_0',
        output='screen',
        parameters=[
            {'image_topic': '/camera/rectified/split_0'},
            {'detection_topic': '/yolo/detections_0/depth'},
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
            {'detection_topic': '/yolo/detections_1/depth'},
            {'output_topic': '/camera/yolo_overlay_1'}
        ]
    )
    
    return launch.LaunchDescription([
        camera_node,
        camera_splitter_node,
        delayed_manual_focus_node,
        sync_capture_node,
        camera_rectification_node,
        stereo_depth_node,
        yolov8_detection_0,
        # yolov8_detection_1,
        object_depth_fusion_node_0,
        # object_depth_fusion_node_1,
        overlay_0,
        # overlay_1,        
    ])