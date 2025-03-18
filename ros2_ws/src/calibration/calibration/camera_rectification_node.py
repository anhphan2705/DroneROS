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

class RectificationNode(Node):
    def __init__(self):
        super().__init__('rectification_node')
        self.br = CvBridge()
        
        # Define the absolute path to the calibration parameters directory
        package_share_dir = get_package_share_directory('calibration')
        self.calibration_dir = os.path.join(package_share_dir, "calibrated_params")
                
        # Dictionaries to store calibration data for each pair.
        self.rectification_maps = {}  # Keys: pair_id, Value: {"left": (map_x, map_y), "right": (map_x, map_y)}
        self.calibrated_pairs = {0: False, 1: False}

        # Load calibration for stereo pair 0 and pair 1.
        success0 = self.apply_latest_calibration(0)
        success1 = self.apply_latest_calibration(1)

        # If neither stereo pair was successfully calibrated, raise Error
        if not success0 and not success1:
            raise RuntimeError("No valid calibration data found for any stereo pair. Exiting.")
        
        # Initialize image_publishers dictionary before setting up subscriptions
        self.image_publishers = {}
        
        # Mapping of topics to the corresponding stereo pair and side.
        self.topic_to_pair = {
            '/camera/image_raw/split_0': (0, "left"),
            '/camera/image_raw/split_1': (0, "right"),
            '/camera/image_raw/split_2': (1, "left"),
            '/camera/image_raw/split_3': (1, "right"),
        }

        # Create image_publishers for rectified images (one for each input topic)
        for topic in self.topic_to_pair.keys():
            # Subscribe to each raw image topic.
            self.create_subscription(
                Image, topic,
                partial(self.image_callback, topic=topic), 10
            )
            # Create a publisher for the rectified version. Here we publish on a topic with a similar name.
            rect_topic = f"/camera/rectified/{topic.split('/')[-1]}"
            self.image_publishers[topic] = self.create_publisher(Image, rect_topic, 10)
            self.get_logger().info(f"Subscribed to {topic} and publishing rectified images on {rect_topic}")
            
        self.get_logger().info(f"RectificationNode initialized and ready.")

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
        
        # Read calibration parameters
        camera_matrix_left = cv_file.getNode("camera_matrix_left").mat()
        dist_coeffs_left = cv_file.getNode("dist_coeffs_left").mat()
        camera_matrix_right = cv_file.getNode("camera_matrix_right").mat()
        dist_coeffs_right = cv_file.getNode("dist_coeffs_right").mat()
        rotation_matrix = cv_file.getNode("rotation_matrix").mat()
        translation_vector = cv_file.getNode("translation_vector").mat()
        rectification_matrix_left = cv_file.getNode("rectification_matrix_left").mat()
        rectification_matrix_right = cv_file.getNode("rectification_matrix_right").mat()
        projection_matrix_left = cv_file.getNode("projection_matrix_left").mat()
        projection_matrix_right = cv_file.getNode("projection_matrix_right").mat()
        disparity_to_depth_map = cv_file.getNode("disparity_to_depth_map").mat()

        # Read rectification maps
        stereo_map_left_x = cv_file.getNode("stereo_map_left_x").mat()
        stereo_map_left_y = cv_file.getNode("stereo_map_left_y").mat()
        stereo_map_right_x = cv_file.getNode("stereo_map_right_x").mat()
        stereo_map_right_y = cv_file.getNode("stereo_map_right_y").mat()

        cv_file.release()

        # Validate calibration data
        if camera_matrix_left is None or camera_matrix_right is None:
            self.get_logger().warn(f"Calibration file {latest_file} is incomplete. Recalibration needed.")
            return False

        # Apply rectification maps for this stereo pair.
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
        # Get the pair id and side from the mapping.
        pair_side = self.topic_to_pair.get(topic, None)
        if pair_side is None:
            self.get_logger().warn(f"Received image on unknown topic: {topic}")
            return
        pair_id, side = pair_side
        
        # Check if calibration for this pair is available.
        if not self.calibrated_pairs.get(pair_id, False):
            self.get_logger().warn(f"Calibration not available for pair {pair_id}. Skipping rectification.")
            return
        
        # Convert the ROS image message to an OpenCV image.
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            return
        
        # Retrieve the rectification maps for this stereo pair and side.
        rect_map = self.rectification_maps[pair_id].get(side, None)
        if rect_map is None:
            self.get_logger().warn(f"No rectification map for {side} side of pair {pair_id}.")
            return
        
        map_x, map_y = rect_map
        # Apply rectification using cv2.remap.
        rectified_image = cv2.remap(cv_image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        # Convert the rectified OpenCV image back to a ROS image message.
        rect_msg = self.br.cv2_to_imgmsg(rectified_image, encoding='bgr8')
        rect_msg.header = msg.header  # Preserve the original header (stamp, frame_id, etc.)
        
        # Publish the rectified image.
        publisher = self.image_publishers.get(topic, None)
        if publisher is not None:
            publisher.publish(rect_msg)
            self.get_logger().debug(f"Published rectified image for topic {topic}")
        else:
            self.get_logger().warn(f"No publisher found for topic {topic}")

def main(args=None):
    rclpy.init(args=args)
    try:
        node = RectificationNode()
        rclpy.spin(node)
    except RuntimeError as e:
        rclpy.get_logger("rectification_node").error(str(e))
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()