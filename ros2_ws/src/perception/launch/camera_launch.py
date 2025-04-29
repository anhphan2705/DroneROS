import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
import os

calib_pkg_share = get_package_share_directory('perception')

calib_file_0 = os.path.join(
    calib_pkg_share,
    'calibrated_params',
    'stereo_calibration_params_pair_0_2025-04-28_16-39-56.yml'
)

calib_file_1 = os.path.join(
    calib_pkg_share,
    'calibrated_params',
    'stereo_calibration_params_pair_1_2025-03-18_02-00-20.yml'
)
    
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
        package='perception',
        executable='camera_rectification_node',
        name='camera_rectification_node',
        output='screen',
    )

    stereo_depth_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='stereo_depth_node',
        name='stereo_depth_node',
        output='screen',
        parameters=[
            {'sub_left': '/camera/rectified/split_0'},
            {'sub_right': '/camera/rectified/split_1'},
            {'depth_publisher': '/camera/depth_map_0'},
            {'horizontal_fov_deg': 66.0},
            {'baseline_m': 0.05},
            
        ]
    )
    
    # stereo_depth_node_1 = launch_ros.actions.Node(
    #     package='perception',
    #     executable='stereo_depth_node',
    #     name='stereo_depth_node',
    #     output='screen',
    #     parameters=[
    #         {'sub_left': '/camera/rectified/split_2'},
    #         {'sub_right': '/camera/rectified/split_3'},
    #         {'depth_publisher': '/camera/depth_map_1'},
    #         {'horizontal_fov_deg': 66.0},
    #         {'baseline_m': 0.05},
            
    #     ]
    # )

    yolov8_detection_0 = launch_ros.actions.Node(
        package='perception',
        executable='object_detection_node',
        name='yolo_detection_0',
        output='screen',
        parameters=[
            {'model_path': 'yolov8s_fp16.engine'},
            {'image_topic': '/camera/rectified/split_0'},
            {'detection_topic': '/yolo/detections_0'}
        ]
    )

    # yolov8_detection_1 = launch_ros.actions.Node(
    #     package='perception',
    #     executable='object_detection_node',
    #     name='yolo_detection_1',
    #     output='screen',
    #     parameters=[
    #         {'model_path': 'yolov8s_fp16.engine'},
    #         {'image_topic': '/camera/rectified/split_2'},
    #         {'detection_topic': '/yolo/detections_1'}
    #     ]
    # )

    # overlay_0 = launch_ros.actions.Node(
    #     package='perception',
    #     executable='bbox_overlay_node',
    #     name='bbox_overlay_0',
    #     output='screen',
    #     parameters=[
    #         {'image_topic': '/camera/rectified/split_0'},
    #         {'detection_topic': '/yolo/detections_0/depth'},
    #         {'output_topic': '/camera/yolo_overlay_0'}
    #     ]
    # )

    # overlay_1 = launch_ros.actions.Node(
    #     package='perception',
    #     executable='bbox_overlay_node',
    #     name='bbox_overlay_1',
    #     output='screen',
    #     parameters=[
    #         {'image_topic': '/camera/rectified/split_2'},
    #         {'detection_topic': '/yolo/detections_1/depth'},
    #         {'output_topic': '/camera/yolo_overlay_1'}
    #     ]
    # )

    byte_track_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='byte_track_node',
        name='byte_track_node_0',
        output='screen',
        parameters=[
            {'input_topic': '/yolo/detections_0'},
            {'output_topic': '/yolo/detections_0/tracked'},
            {'image_topic': '/camera/rectified/split_0'},
            {'horizontal_fov_deg': 66.0},
            {'vertical_fov_deg': 49.5},
        ]
    )
    
    # byte_track_node_1 = launch_ros.actions.Node(
    #     package='perception',
    #     executable='byte_track_node',
    #     name='byte_track_node_1',
    #     output='screen',
    #     parameters=[
    #         {'input_topic': '/yolo/detections_1'},
    #         {'output_topic': '/yolo/detections_1/tracked'},
    #         {'image_topic': '/camera/rectified/split_2'},
    #         {'horizontal_fov_deg': 66.0},
    #         {'vertical_fov_deg': 49.5},
    #     ]
    # )
    
    classification_node_id11 = launch_ros.actions.Node(
        package='perception',
        executable='classification_node',
        name='classification_node',
        output='screen',
        parameters=[
            {'tracked_topic': '/yolo/detections_0/tracked'},
            {'image_topic': '/camera/rectified/split_0'},
            {'classifier_model_path': 'sign_classification_model.pt'},
            {'input_size': [320, 320]},  # [0, 0] for dynamic crop sizes
            {'class_ids_to_classify': [11]},  # e.g. [1, 2, 3]
            {'classification_topic': '/yolo/detections_0/tracked/classified'},
        ]
    )
    
    traffic_light_classification = launch_ros.actions.Node(
        package='perception',
        executable='traffic_light_classification_node',
        name='traffic_light_classification',
        output='screen',
        parameters=[
            {'tracked_topic': '/yolo/detections_0/tracked/classified'},
            {'image_topic': '/camera/rectified/split_0'},
            {'input_size': [0, 0]},
            {'class_ids_to_classify': [9]},  # e.g. [1, 2, 3]
            {'classification_topic': '/yolo/detections_0/tracked/classified/light'},
        ]
    )
    
    object_depth_fusion_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='object_depth_fusion_node',
        name='object_depth_fusion_node_0',
        output='screen',
        parameters=[
            {'detection_topic': '/yolo/detections_0/tracked/classified/light'},
            {'depth_topic': '/camera/depth_map_0'},
            {'output_topic': '/yolo/detections_0/tracked/classified/light/depth'}
        ]
    )
    
    # object_depth_fusion_node_1 = launch_ros.actions.Node(
    #     package='perception',
    #     executable='object_depth_fusion_node',
    #     name='object_depth_fusion_node',
    #     output='screen',
    #     parameters=[
    #         {'detection_topic': '/yolo/detections_1'},
    #         {'depth_topic': '/camera/depth_map_1'},
    #         {'output_topic': '/yolo/detections_1/depth'}
    #     ]
    # )
    
    speed_estimator_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='speed_estimator_node',
        name='speed_estimator_node_0',
        output='screen',
        parameters=[
            {'subscribe_topic': '/yolo/detections_0/tracked/classified/light/depth'},
            {'calibration_file': calib_file_0},
            {'publish_topic': '/yolo/detections_0/tracked/classified/light/depth/speed'}
        ]
    )
    
    # speed_estimator_node_0 = launch_ros.actions.Node(
    #     package='perception',
    #     executable='speed_estimator_node',
    #     name='speed_estimator_node_0',
    #     output='screen',
    #     parameters=[
    #         {'subscribe_topic': '/yolo/detections_1/tracked/classified/light/depth'},
    #         {'calibration_file': calib_file_1},
    #         {'publish_topic': '/yolo/detections_1/tracked/classified/light/depth/speed'}
    #     ]
    # )

    id_mapper_node = launch_ros.actions.Node(
        package='perception',
        executable='id_mapper_node',
        name='id_mapper_node',
        output='screen',
        parameters=[
            {'mapping_file': 'id_map.yaml'},
            {'tracked_topic': '/yolo/detections_0/tracked/classified/light/depth/speed'},
            {'map_topic': '/yolo/detections_0/tracked/classified/light/depth/speed/mapped'}
        ]
    )

    tracked_overlay_0 = launch_ros.actions.Node(
        package='perception',
        executable='tracked_bbox_overlay_node',
        name='tracked_overlay_0',
        output='screen',
        parameters=[
            {'image_topic': '/camera/rectified/split_0'},
            {'tracked_topic': '/yolo/detections_0/tracked/classified/light/depth/speed/mapped'},
            {'output_topic': '/perception_img_visualizer_0'}
        ]
    )
     
    # tracked_overlay_1 = launch_ros.actions.Node(
    #     package='perception',
    #     executable='tracked_bbox_overlay_node',
    #     name='tracked_overlay_1',
    #     output='screen',
    #     parameters=[
    #         {'image_topic': '/camera/rectified/split_1'},
    #         {'tracked_topic': '/yolo/detections_1/depth/tracked/classified/light'},
    #         {'output_topic': '/camera/yolo_overlay_tracked_1'}
    #     ]
    # )

    return launch.LaunchDescription([
        camera_node,
        camera_splitter_node,
        delayed_manual_focus_node,
        sync_capture_node,
        camera_rectification_node,
        stereo_depth_node_0,
        # stereo_depth_node_1,
        yolov8_detection_0,
        # yolov8_detection_1,
        # overlay_0,
        # overlay_1,
        # object_depth_fusion_tracked_node_0,
        # object_depth_fusion_tracked_node_1,   
        byte_track_node_0,
        # byte_track_node_1,
        classification_node_id11,
        traffic_light_classification,
        object_depth_fusion_node_0,
        # object_depth_fusion_node_1,
        speed_estimator_node_0,
        # speed_estimator_node_1,
        id_mapper_node,
        tracked_overlay_0,
        # tracked_overlay_1,
    ])