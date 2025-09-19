import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
import os

calib_pkg_share = get_package_share_directory('perception')

calib_file_0 = os.path.join(
    calib_pkg_share,
    'calibrated_params',
    'stereo_calibration_params_pair_0_2025-07-11_19-45-40.yml'
)

calib_file_1 = os.path.join(
    calib_pkg_share,
    'calibrated_params',
    'stereo_calibration_params_pair_1_2025-07-11_19-46-08.yml'
)
    
def generate_launch_description():
    camera_gpu_node = launch_ros.actions.Node(
        package='perception',
        executable='camera_gpu_node',
        name='camera_gpu_node',
        output='screen',
        parameters=[
            {'width': 1920},
            {'height': 1080},
            {'fps': 60},
            {'udp_host': '192.168.0.254'},
            {'udp_port': 5600},
            {'bitrate_kbps': 2500},
            {'calibration_file_0': calib_file_0},
            {'calibration_file_1': calib_file_1},
        ]
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
        output='screen',
        parameters=[
            {"camera_topic": "/camera/image_raw"},
        ]
    )

    delayed_manual_focus_node = launch.actions.TimerAction(
        period=10.0,
        actions=[manual_focus_node]
    )

    stereo_depth_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='stereo_depth_node',
        name='stereo_depth_node_0',
        output='screen',
        parameters=[
            {'sub_left': '/camera0/rectified'},
            {'sub_right': '/camera1/rectified'},
            {'depth_publisher': '/camera0/depth_map'},
            {'calibration_file': calib_file_0},
        ]
    )
    
    stereo_depth_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='stereo_depth_node',
        name='stereo_depth_node_1',
        output='screen',
        parameters=[
            {'sub_left': '/camera2/rectified'},
            {'sub_right': '/camera3/rectified'},
            {'depth_publisher': '/camera2/depth_map'},
            {'calibration_file': calib_file_1},
        ]
    )
    
    yolov8_detection_0 = launch_ros.actions.Node(
        package='perception',
        executable='object_detection_node',
        name='yolo_detection_0',
        output='screen',
        parameters=[
            {'model_path': 'yolov8s_fp16.engine'},
            {'image_topic': '/camera0/rectified'},
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
            {'image_topic': '/camera2/rectified'},
            {'detection_topic': '/yolo/detections_1'}
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
            {'image_topic': '/camera0/rectified'},
            {'calibration_file': calib_file_0},

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
            {'image_topic': '/camera2/rectified'},
            {'calibration_file': calib_file_1},
        ]
    )
    
    classification_node_id11_0 = launch_ros.actions.Node(
        package='perception',
        executable='classification_node',
        name='classification_node_0',
        output='screen',
        parameters=[
            {'tracked_topic': '/yolo/detections_0/tracked'},
            {'image_topic': '/camera0/rectified'},
            {'classifier_model_path': 'sign_classification_model.pt'},
            {'input_size': [320, 320]},  # [0, 0] for dynamic crop sizes
            {'class_ids_to_classify': [11]},  # e.g. [1, 2, 3]
            {'classification_topic': '/yolo/detections_0/tracked/classified'},
        ]
    )
    
    classification_node_id11_1 = launch_ros.actions.Node(
        package='perception',
        executable='classification_node',
        name='classification_node_1',
        output='screen',
        parameters=[
            {'tracked_topic': '/yolo/detections_1/tracked'},
            {'image_topic': '/camera2/rectified'},
            {'classifier_model_path': 'sign_classification_model.pt'},
            {'input_size': [320, 320]},  # [0, 0] for dynamic crop sizes
            {'class_ids_to_classify': [11]},  # e.g. [1, 2, 3]
            {'classification_topic': '/yolo/detections_1/tracked/classified'},
        ]
    )
    
    object_depth_fusion_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='object_depth_fusion_node',
        name='object_depth_fusion_node_0',
        output='screen',
        parameters=[
            {'detection_topic': '/yolo/detections_0/tracked/classified'},
            {'depth_topic': '/camera0/depth_map'},
            {'output_topic': '/yolo/detections_0/tracked/classified/depth'}
        ]
    )
    
    object_depth_fusion_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='object_depth_fusion_node',
        name='object_depth_fusion_node_1',
        output='screen',
        parameters=[
            {'detection_topic': '/yolo/detections_1/tracked/classified'},
            {'depth_topic': '/camera2/depth_map'},
            {'output_topic': '/yolo/detections_1/tracked/classified/depth'}
        ]
    )
    
    speed_estimator_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='speed_estimator_node',
        name='speed_estimator_node_0',
        output='screen',
        parameters=[
            {'subscribe_topic': '/yolo/detections_0/tracked/classified/depth'},
            {'calibration_file': calib_file_0},
            {'publish_topic': '/yolo/detections_0/tracked/classified/depth/speed'}
        ]
    )
    
    speed_estimator_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='speed_estimator_node',
        name='speed_estimator_node_1',
        output='screen',
        parameters=[
            {'subscribe_topic': '/yolo/detections_1/tracked/classified/depth'},
            {'calibration_file': calib_file_1},
            {'publish_topic': '/yolo/detections_1/tracked/classified/depth/speed'}
        ]
    )

    id_mapper_node_0 = launch_ros.actions.Node(
        package='perception',
        executable='id_mapper_node',
        name='id_mapper_node_0',
        output='screen',
        parameters=[
            {'mapping_file': 'id_map.yaml'},
            {'tracked_topic': '/yolo/detections_0/tracked/classified/depth/speed'},
            {'map_topic': '/yolo/detections_0/tracked/classified/depth/speed/mapped'}
        ]
    )
    
    id_mapper_node_1 = launch_ros.actions.Node(
        package='perception',
        executable='id_mapper_node',
        name='id_mapper_node_1',
        output='screen',
        parameters=[
            {'mapping_file': 'coco_id_map.yaml'},
            {'tracked_topic': '/yolo/detections_1/tracked/classified/depth/speed'},
            {'map_topic': '/yolo/detections_1/tracked/classified/depth/speed/mapped'}
        ]
    )

    tracked_overlay_0 = launch_ros.actions.Node(
        package='perception',
        executable='tracked_bbox_overlay_node',
        name='tracked_overlay_0',
        output='screen',
        parameters=[
            {'image_topic': '/camera0/rectified'},
            {'tracked_topic': '/yolo/detections_0/tracked/classified/depth/speed/mapped'},
            {'output_topic': '/perception_img_visualizer_0'}
        ]
    )
     
    tracked_overlay_1 = launch_ros.actions.Node(
        package='perception',
        executable='tracked_bbox_overlay_node',
        name='tracked_overlay_1',
        output='screen',
        parameters=[
            {'image_topic': '/camera2/rectified'},
            {'tracked_topic': '/yolo/detections_1/tracked/classified/depth/speed/mapped'},
            {'output_topic': '/perception_img_visualizer_1'}
        ]
    )

    return launch.LaunchDescription([
        camera_gpu_node,
        delayed_manual_focus_node,
        sync_capture_node,
        # stereo_depth_node_0,
        # stereo_depth_node_1,
        yolov8_detection_0,
        # yolov8_detection_1,
        byte_track_node_0,
        # byte_track_node_1,
        classification_node_id11_0,
        # classification_node_id11_1,
        object_depth_fusion_node_0,
        # object_depth_fusion_node_1,
        speed_estimator_node_0,
        # speed_estimator_node_1,
        id_mapper_node_0,
        # id_mapper_node_1,
        tracked_overlay_0,
        # tracked_overlay_1,
    ])
