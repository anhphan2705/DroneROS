import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
import os

calib_pkg_share = get_package_share_directory('perception')

calib_file_0 = os.path.join(
    calib_pkg_share,
    'calibrated_params',
    'stereo_calibration_params_pair_0_2025-06-15_21-14-08.yml'
)

calib_file_1 = os.path.join(
    calib_pkg_share,
    'calibrated_params',
    'stereo_calibration_params_pair_1_2025-06-15_21-20-05.yml'
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

    camera_rectification_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='camera_rectification_node',
        name='camera_rectification_node_0',
        parameters=[
            {'left_image_topic': '/camera/image_raw/split_1'},
            {'right_image_topic': '/camera/image_raw/split_0'},
            {'calibration_file': calib_file_0},
            {'output_prefix': '/camera/rectified_0'}
        ]
    )

    camera_rectification_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='camera_rectification_node',
        name='camera_rectification_node_1',
        parameters=[
            {'left_image_topic': '/camera/image_raw/split_3'},
            {'right_image_topic': '/camera/image_raw/split_2'},
            {'calibration_file': calib_file_1},
            {'output_prefix': '/camera/rectified_1'}
        ]
    )

    stereo_depth_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='stereo_depth_node',
        name='stereo_depth_node_0',
        output='screen',
        parameters=[
            {'sub_left': '/camera/rectified_0/left'},
            {'sub_right': '/camera/rectified_0/right'},
            {'depth_publisher': '/camera/rectified_0/depth_map'},
            {'calibration_file': calib_file_0},
        ]
    )
    
    stereo_depth_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='stereo_depth_node',
        name='stereo_depth_node_1',
        output='screen',
        parameters=[
            {'sub_left': '/camera/rectified_1/left'},
            {'sub_right': '/camera/rectified_1/right'},
            {'depth_publisher': '/camera/rectified_1/depth_map'},
            {'calibration_file': calib_file_1},
        ]
    )

    depth_viz_0 = launch_ros.actions.Node(
        package='perception',
        executable='depth_visualizer_node',
        name='depth_visualizer_0',
        output='screen',
        parameters=[
            {'depth_topic': '/camera/rectified_0/depth_map'},
            {'output_topic': '/camera/rectified_0/depth_map/depth_vis'},
        ]
    )
    
    depth_viz_1 = launch_ros.actions.Node(
        package='perception',
        executable='depth_visualizer_node',
        name='depth_visualizer_1',
        output='screen',
        parameters=[
            {'depth_topic': '/camera/rectified_1/depth_map'},
            {'output_topic': '/camera/rectified_1/depth_map/depth_vis'},
        ]
    )
    
    yolov8_detection_0 = launch_ros.actions.Node(
        package='perception',
        executable='object_detection_node',
        name='yolo_detection_0',
        output='screen',
        parameters=[
            {'model_path': 'yolov8s_fp16.engine'},
            {'image_topic': '/camera/rectified_0/left'},
            {'detection_topic': '/yolo/detections_0'}
        ]
    )

    yolov8_detection_1 = launch_ros.actions.Node(
        package='perception',
        executable='object_detection_node',
        name='yolo_detection_1',
        output='screen',
        parameters=[
            {'model_path': 'yolov8s_fp16.engine'},
            {'image_topic': '/camera/rectified_1/left'},
            {'detection_topic': '/yolo/detections_1'}
        ]
    )

    # overlay_0 = launch_ros.actions.Node(
    #     package='perception',
    #     executable='bbox_overlay_node',
    #     name='bbox_overlay_0',
    #     output='screen',
    #     parameters=[
    #         {'image_topic': '/camera/rectified_0/left'},
    #         {'detection_topic': '/yolo/detections_0'},
    #         {'output_topic': '/camera/yolo_overlay_0'}
    #     ]
    # )

    overlay_1 = launch_ros.actions.Node(
        package='perception',
        executable='bbox_overlay_node',
        name='bbox_overlay_1',
        output='screen',
        parameters=[
            {'image_topic': '/camera/rectified_1/left'},
            {'detection_topic': '/yolo/detections_1'},
            {'output_topic': '/camera/yolo_overlay_1'}
        ]
    )

    byte_track_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='byte_track_node',
        name='byte_track_node_0',
        output='screen',
        parameters=[
            {'input_topic': '/yolo/detections_0'},
            {'output_topic': '/yolo/detections_0/tracked'},
            {'image_topic': '/camera/rectified_0/left'},
            {'horizontal_fov_deg': 66.0},
            {'vertical_fov_deg': 49.5},
        ]
    )
    
    byte_track_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='byte_track_node',
        name='byte_track_node_1',
        output='screen',
        parameters=[
            {'input_topic': '/yolo/detections_1'},
            {'output_topic': '/yolo/detections_1/tracked'},
            {'image_topic': '/camera/rectified_1/left'},
            {'horizontal_fov_deg': 66.0},
            {'vertical_fov_deg': 49.5},
        ]
    )
    
    classification_node_id11_0 = launch_ros.actions.Node(
        package='perception',
        executable='classification_node',
        name='classification_node_0',
        output='screen',
        parameters=[
            {'tracked_topic': '/yolo/detections_0/tracked'},
            {'image_topic': '/camera/rectified_0/left'},
            {'classifier_model_path': 'sign_classification_model.pt'},
            {'input_size': [320, 320]},  # [0, 0] for dynamic crop sizes
            {'class_ids_to_classify': [11]},  # e.g. [1, 2, 3]
            {'classification_topic': '/yolo/detections_0/tracked/classified'},
        ]
    )
    
    traffic_light_classification_0 = launch_ros.actions.Node(
        package='perception',
        executable='traffic_light_classification_node',
        name='traffic_light_classification_0',
        output='screen',
        parameters=[
            {'tracked_topic': '/yolo/detections_0/tracked/classified'},
            {'image_topic': '/camera/rectified_0/left'},
            {'input_size': [0, 0]},
            {'class_ids_to_classify': [9]},  # e.g. [1, 2, 3]
            {'classification_topic': '/yolo/detections_0/tracked/classified/light'},
        ]
    )
    
    classification_node_id11_1 = launch_ros.actions.Node(
        package='perception',
        executable='classification_node',
        name='classification_node_1',
        output='screen',
        parameters=[
            {'tracked_topic': '/yolo/detections_1/tracked'},
            {'image_topic': '/camera/rectified_1/left'},
            {'classifier_model_path': 'sign_classification_model.pt'},
            {'input_size': [320, 320]},  # [0, 0] for dynamic crop sizes
            {'class_ids_to_classify': [11]},  # e.g. [1, 2, 3]
            {'classification_topic': '/yolo/detections_1/tracked/classified'},
        ]
    )
    
    traffic_light_classification_1 = launch_ros.actions.Node(
        package='perception',
        executable='traffic_light_classification_node',
        name='traffic_light_classification_1',
        output='screen',
        parameters=[
            {'tracked_topic': '/yolo/detections_1/tracked/classified'},
            {'image_topic': '/camera/rectified_1/left'},
            {'input_size': [0, 0]},
            {'class_ids_to_classify': [9]},  # e.g. [1, 2, 3]
            {'classification_topic': '/yolo/detections_1/tracked/classified/light'},
        ]
    )
    
    object_depth_fusion_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='object_depth_fusion_node',
        name='object_depth_fusion_node_0',
        output='screen',
        parameters=[
            {'detection_topic': '/yolo/detections_0/tracked/classified/light'},
            {'depth_topic': '/camera/rectified_0/depth_map'},
            {'output_topic': '/yolo/detections_0/tracked/classified/light/depth'}
        ]
    )
    
    object_depth_fusion_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='object_depth_fusion_node',
        name='object_depth_fusion_node_1',
        output='screen',
        parameters=[
            {'detection_topic': '/yolo/detections_1/tracked/classified/light'},
            {'depth_topic': '/camera/rectified_1/depth_map'},
            {'output_topic': '/yolo/detections_1/tracked/classified/light/depth'}
        ]
    )
    
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
    
    speed_estimator_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='speed_estimator_node',
        name='speed_estimator_node_1',
        output='screen',
        parameters=[
            {'subscribe_topic': '/yolo/detections_1/tracked/classified/light/depth'},
            {'calibration_file': calib_file_1},
            {'publish_topic': '/yolo/detections_1/tracked/classified/light/depth/speed'}
        ]
    )

    id_mapper_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='id_mapper_node',
        name='id_mapper_node_0',
        output='screen',
        parameters=[
            {'mapping_file': 'id_map.yaml'},
            {'tracked_topic': '/yolo/detections_0/tracked/classified/light/depth/speed'},
            {'map_topic': '/yolo/detections_0/tracked/classified/light/depth/speed/mapped'}
        ]
    )
    
    id_mapper_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='id_mapper_node',
        name='id_mapper_node_1',
        output='screen',
        parameters=[
            {'mapping_file': 'coco_id_map.yaml'},
            {'tracked_topic': '/yolo/detections_1/tracked/classified/light/depth/speed'},
            {'map_topic': '/yolo/detections_1/tracked/classified/light/depth/speed/mapped'}
        ]
    )

    tracked_overlay_0 = launch_ros.actions.Node(
        package='perception',
        executable='tracked_bbox_overlay_node',
        name='tracked_overlay_0',
        output='screen',
        parameters=[
            {'image_topic': '/camera/rectified_0/left'},
            {'tracked_topic': '/yolo/detections_0/tracked/classified/light/depth/speed/mapped'},
            {'output_topic': '/perception_img_visualizer_0'}
        ]
    )
     
    tracked_overlay_1 = launch_ros.actions.Node(
        package='perception',
        executable='tracked_bbox_overlay_node',
        name='tracked_overlay_1',
        output='screen',
        parameters=[
            {'image_topic': '/camera/rectified_1/left'},
            {'tracked_topic': '/yolo/detections_1/tracked/classified/light/depth/speed/mapped'},
            {'output_topic': '/perception_img_visualizer_1'}
        ]
    )

    return launch.LaunchDescription([
        camera_node,
        camera_splitter_node,
        delayed_manual_focus_node,
        sync_capture_node,
        # camera_rectification_node_0,
        camera_rectification_node_1,
        # stereo_depth_node_0,
        stereo_depth_node_1,
        # depth_viz_0,
        depth_viz_1,
        # yolov8_detection_0,
        yolov8_detection_1,
        # overlay_0,
        overlay_1,
        # byte_track_node_0,
        byte_track_node_1,
        # classification_node_id11_0,
        # traffic_light_classification_0,
        classification_node_id11_1,
        traffic_light_classification_1,
        # object_depth_fusion_node_0,
        object_depth_fusion_node_1,
        # speed_estimator_node_0,
        speed_estimator_node_1,
        # id_mapper_node_0,
        id_mapper_node_1,
        # tracked_overlay_0,
        tracked_overlay_1,
    ])
