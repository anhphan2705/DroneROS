#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import glob
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from functools import partial
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor

class RectificationNode(Node):
    def __init__(self):
        super().__init__('camera_rectification_node')
        self.br = CvBridge()
        
        # Define the absolute path to the calibration parameters directory
        package_share_dir = get_package_share_directory('calibration')
        self.calibration_dir = os.path.join(package_share_dir, "calibrated_params")

        # Dictionaries to store calibration data
        self.rectification_maps = {}
        self.calibrated_pairs = {0: False, 1: False}

        # Load calibration for both stereo pairs
        success0 = self.apply_latest_calibration(0)
        success1 = self.apply_latest_calibration(1)

        if not success0 and not success1:
            raise RuntimeError("No valid calibration data found for any stereo pair. Exiting.")

        # Initialize image publishers
        self.image_publishers = {}

        # Mapping of topics to stereo pairs
        self.topic_to_pair = {
            '/camera/image_raw/split_0': (0, "left"),
            '/camera/image_raw/split_1': (0, "right"),
            '/camera/image_raw/split_2': (1, "left"),
            '/camera/image_raw/split_3': (1, "right"),
        }

        # Subscribe and create publishers
        for topic in self.topic_to_pair.keys():
            self.create_subscription(
                Image, topic,
                partial(self.image_callback, topic=topic), 10
            )
            rect_topic = f"/camera/rectified/{topic.split('/')[-1]}"
            self.image_publishers[topic] = self.create_publisher(Image, rect_topic, 10)
            self.get_logger().info(f"Subscribed to {topic}, publishing rectified images on {rect_topic}")

        self.get_logger().info("RectificationNode initialized!")

    def apply_latest_calibration(self, pair_id):
        """ Loads and applies the latest calibration file if available. """
        search_pattern = os.path.join(self.calibration_dir, f"stereo_calibration_params_pair_{pair_id}_*.yml")
        files = sorted(glob.glob(search_pattern), reverse=True)

        if not files:
            self.get_logger().warn(f"No calibration file found for stereo pair {pair_id}. Please recalibrate.")
            return False
        
        latest_file = files[0]
        self.get_logger().info(f"Loading calibration file: {latest_file}")

        cv_file = cv2.FileStorage(latest_file, cv2.FILE_STORAGE_READ)
        if not cv_file.isOpened():
            self.get_logger().error(f"Failed to open calibration file: {latest_file}")
            return False

        # Read camera matrices & distortion coefficients
        frame_size = cv_file.getNode("frame_size").mat()
        camera_matrix_left = cv_file.getNode("camera_matrix_left").mat()
        dist_coeffs_left = cv_file.getNode("dist_coeffs_left").mat()
        camera_matrix_right = cv_file.getNode("camera_matrix_right").mat()
        dist_coeffs_right = cv_file.getNode("dist_coeffs_right").mat()
        rectification_matrix_left = cv_file.getNode("rectification_matrix_left").mat()
        rectification_matrix_right = cv_file.getNode("rectification_matrix_right").mat()
        projection_matrix_left = cv_file.getNode("projection_matrix_left").mat()
        projection_matrix_right = cv_file.getNode("projection_matrix_right").mat()

        cv_file.release()

        # Validate calibration data
        if camera_matrix_left is None or camera_matrix_right is None:
            self.get_logger().warn(f"Calibration file {latest_file} is incomplete. Recalibration needed.")
            return False

        width, height = int(frame_size[0, 0]), int(frame_size[1, 0])

        stereo_map_left_x, stereo_map_left_y = cv2.initUndistortRectifyMap(
            camera_matrix_left, dist_coeffs_left, rectification_matrix_left,
            projection_matrix_left, (width, height), cv2.CV_32FC1
        )

        stereo_map_right_x, stereo_map_right_y = cv2.initUndistortRectifyMap(
            camera_matrix_right, dist_coeffs_right, rectification_matrix_right,
            projection_matrix_right, (width, height), cv2.CV_32FC1
        )

        # Store rectification maps for fast access
        self.rectification_maps[pair_id] = {
            "left": (stereo_map_left_x, stereo_map_left_y),
            "right": (stereo_map_right_x, stereo_map_right_y)
        }
        
        self.calibrated_pairs[pair_id] = True
        self.get_logger().info(f"Applied calibration from {latest_file} for stereo pair {pair_id}.")
        return True

    def image_callback(self, msg, topic):
        """
        Callback for raw images. Determines which stereo pair and side the image belongs to,
        applies the corresponding rectification maps, and publishes the rectified image.
        """
        pair_side = self.topic_to_pair.get(topic, None)
        if pair_side is None:
            self.get_logger().warn(f"Received image on unknown topic: {topic}")
            return

        pair_id, side = pair_side

        if not self.calibrated_pairs.get(pair_id, False):
            self.get_logger().warn(f"Calibration not available for pair {pair_id}. Skipping rectification.")
            return

        try:
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            return
        
        # Convert to UMat for hardware acceleration
        cv_image = cv2.UMat(cv_image)

        # Retrieve the precomputed rectification maps
        map_x, map_y = self.rectification_maps[pair_id][side]

        # Apply rectification
        rectified_image = cv2.remap(cv_image, map_x, map_y, interpolation=cv2.INTER_LINEAR).get()
        # rectified_image = cv2.remap(cv_image, map_x, map_y, interpolation=cv2.INTER_NEAREST).get()

        # Convert rectified image back to ROS 2 Image message
        rect_msg = self.br.cv2_to_imgmsg(rectified_image, encoding='bgr8')
        rect_msg.header = msg.header

        publisher = self.image_publishers.get(topic, None)
        if publisher:
            publisher.publish(rect_msg)

def main(args=None):
    rclpy.init(args=args)
    
    # Use MultiThreadedExecutor for parallel processing
    executor = MultiThreadedExecutor()
    node = RectificationNode()
    executor.add_node(node)

    try:
        executor.spin()
    except RuntimeError as e:
        rclpy.get_logger("rectification_node").error(str(e))
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()