import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.clock import Clock
from sensor_msgs.msg import Image
from msgs.srv import CameraCalibrationRequest
from cv_bridge import CvBridge
import cv2
import yaml
import os
import glob
import numpy as np
from datetime import datetime


class StereoCalibrationNode(Node):
    def __init__(self):
        super().__init__('stereo_calibration_node')

        self.bridge = CvBridge()
        self.calibrated_pairs = {1: False, 2: False}  # Track calibration status
        self.min_images = 20  # Minimum images required for calibration
        self.calibration_dir = "src/calibration/calibration/calibration_result"
        self.calibration_in_progress = False
        self.last_capture_time = {1: self.get_clock().now(), 2: self.get_clock().now()}

        os.makedirs(self.calibration_dir, exist_ok=True)

        self.srv = self.create_service(CameraCalibrationRequest, 'start_stereo_calibration', self.calibration_callback)

        self.subs = {
            1: {
                "left": self.create_subscription(Image, "/camera/image_raw/split_0", lambda msg: self.callback_image(msg, 1, "left"), 10),
                "right": self.create_subscription(Image, "/camera/image_raw/split_1", lambda msg: self.callback_image(msg, 1, "right"), 10),
            },
            2: {
                "left": self.create_subscription(Image, "/camera/image_raw/split_2", lambda msg: self.callback_image(msg, 2, "left"), 10),
                "right": self.create_subscription(Image, "/camera/image_raw/split_3", lambda msg: self.callback_image(msg, 2, "right"), 10),
            }
        }

        self.pubs = {
            1: {
                "left": self.create_publisher(Image, "/camera/image_rectified/split_0", 10),
                "right": self.create_publisher(Image, "/camera/image_rectified/split_1", 10),
            },
            2: {
                "left": self.create_publisher(Image, "/camera/image_rectified/split_2", 10),
                "right": self.create_publisher(Image, "/camera/image_rectified/split_3", 10),
            }
        }

        # Image storage
        self.image_buffers = {1: {"left": [], "right": []}, 2: {"left": [], "right": []}}
        self.rectification_maps = {1: None, 2: None}  # Rectification maps for both pairs

        # Load latest calibration if available
        self.load_latest_calibration(1)
        self.load_latest_calibration(2)

    def callback_image(self, msg, pair_id, side):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        if self.calibrated_pairs[pair_id]:
            if self.rectification_maps[pair_id]:
                map1, map2 = self.rectification_maps[pair_id][side]
                img_rectified = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
                self.pubs[pair_id][side].publish(self.bridge.cv2_to_imgmsg(img_rectified, "bgr8"))

    def calibration_callback(self, request, response):
        pair_id = request.pair_id

        if pair_id not in [1, 2]:
            response.success = False
            response.message = "Invalid pair_id. Use 1 or 2."
            return response

        self.calibration_in_progress = True
        self.get_logger().info(f"Starting image capture for stereo pair {pair_id}...")

        for i in range(self.min_images):
            self.get_logger().info(f"Position the chessboard correctly... Capturing image {i+1}/{self.min_images} in:")
            
            for countdown in range(5, 0, -1):
                self.get_logger().info(f"{countdown}...")
                rclpy.spin_once(self, timeout_sec=1.0)

            self.capture_stereo_images(pair_id)

        self.get_logger().info(f"Starting calibration for stereo pair {pair_id}...")

        if len(self.image_buffers[pair_id]["left"]) < self.min_images or len(self.image_buffers[pair_id]["right"]) < self.min_images:
            response.success = False
            response.message = "Not enough images for calibration."
            self.calibration_in_progress = False
            return response

        self.perform_calibration(pair_id)

        self.calibration_in_progress = False
        response.success = True
        response.message = f"Stereo Calibration for pair {pair_id} completed!"
        return response

    def perform_calibration(self, pair_id):
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = f"{self.calibration_dir}/calibration_pair_{pair_id}_{timestamp}.yaml"

        # Compute single-camera calibration parameters and reprojection error
        left_mtx, left_dist, left_error = self.compute_camera_calibration(pair_id, "left")
        right_mtx, right_dist, right_error = self.compute_camera_calibration(pair_id, "right")

        # Compute stereo calibration reprojection error
        stereo_error = (left_error + right_error) / 2

        # Save calibration data
        with open(filename, 'w') as file:
            yaml.dump({
                "left_camera_matrix": left_mtx.tolist(),
                "left_distortion_coefficients": left_dist.tolist(),
                "right_camera_matrix": right_mtx.tolist(),
                "right_distortion_coefficients": right_dist.tolist(),
                "left_reprojection_error": left_error,
                "right_reprojection_error": right_error,
                "stereo_reprojection_error": stereo_error
            }, file)

        self.get_logger().info(f"Calibration for stereo pair {pair_id} saved to {filename}.")

        # ✅ Keep only the last 3 calibration files
        self.manage_calibration_files(pair_id)

    def manage_calibration_files(self, pair_id):
        """ Keeps only the last 3 calibration files, deleting the oldest if needed. """
        files = sorted(glob.glob(f"{self.calibration_dir}/calibration_pair_{pair_id}_*.yaml"))
        if len(files) > 3:
            oldest_file = files[0]
            os.remove(oldest_file)
            self.get_logger().info(f"Deleted oldest calibration file: {oldest_file}")

    def load_latest_calibration(self, pair_id):
        """ Loads the latest calibration file if available. """
        files = sorted(glob.glob(f"{self.calibration_dir}/calibration_pair_{pair_id}_*.yaml"), reverse=True)
        if files:
            latest_file = files[0]
            with open(latest_file, 'r') as file:
                calibration_data = yaml.safe_load(file)
            self.get_logger().info(f"Loaded calibration from {latest_file}.")
            return calibration_data
        else:
            self.get_logger().warn(f"No calibration file found for stereo pair {pair_id}. Please recalibrate.")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = StereoCalibrationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

# HOWTO:
# 1. Run the node:  
#    ros2 run calibration stereo_calibration_node
#
# 2. Trigger calibration for a stereo pair:  
#    ros2 service call /start_stereo_calibration msgs/srv/CameraCalibrationRequest "{pair_id: 1}"
#    ros2 service call /start_stereo_calibration msgs/srv/CameraCalibrationRequest "{pair_id: 2}"
#
# 3. Check stored calibration files:  
#    ls src/calibration/calibration/calibration_result/
#
# 4. View calibration data:  
#    cat src/calibration/calibration/calibration_result/calibration_pair_1_YYYYMMDD_HHMMSS.yaml
#
# Keeps the last 3 calibration files per stereo pair.
# Left & right images are captured together.
