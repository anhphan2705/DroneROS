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
from tools.stereo_camera_calibration import StereoCalibrator


class StereoCalibrationNode(Node):
    def __init__(self):
        super().__init__('stereo_calibration_node')

        self.bridge = CvBridge()
        self.calibrated_pairs = {1: False, 2: False}  # Track calibration status
        self.min_images = 20  # Minimum images required for calibration
        self.calibration_dir = "src/calibration/calibration/calibration_result"
        self.calibration_in_progress = False
        self.last_capture_time = {1: self.get_clock().now(), 2: self.get_clock().now()}
        self.rectification_maps = {1: None, 2: None}
        self.calibration_tool = StereoCalibrator(
                                    use_buffer=True, 
                                    verbose=False, 
                                    chessboard_size=(10, 7), 
                                    square_size=25, 
                                    show_corners= False
                                )

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

        # Load and apply latest calibration if available
        self.apply_latest_calibration(1)
        self.apply_latest_calibration(2)

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
    
    def capture_stereo_images(self, pair_id):
        current_time = self.get_clock().now()
        self.last_capture_time[pair_id] = current_time

        for side in ["left", "right"]:
            if len(self.image_buffers[pair_id][side]) >= self.min_images:
                self.image_buffers[pair_id][side].pop(0)
            
            self.image_buffers[pair_id][side].append(current_time)

        self.get_logger().info(f"Captured stereo image pair for stereo pair {pair_id}.")
    
    def perform_calibration(self, pair_id):
        # Use StereoCalibrator to perform calibration
        self.calibration_tool.load_images_from_buffer(
            left_buffer=self.image_buffers[pair_id]["left"],
            right_buffer=self.image_buffers[pair_id]["right"]
        )
        self.calibration_tool.detect_chessboard_corners_from_buffer()
        self.calibration_tool.calibrate_cameras()
        self.calibration_tool.stereo_calibrate()
        self.calibration_tool.stereo_rectify()
        
        # Save calibration data
        self.calibration_tool.save_parameters(output_base_dir=self.calibration_dir, pair_id=pair_id)
        
        self.get_logger().info(f"Calibration for stereo pair {pair_id} saved.")
        self.manage_calibration_files(pair_id)
        self.apply_latest_calibration(pair_id)

    def apply_latest_calibration(self, pair_id):
        """ Loads and applies the latest calibration file if available. """
        files = sorted(glob.glob(f"{self.calibration_dir}/stereo_calibration_params_pair_{pair_id}_*.yml"), reverse=True)
        
        if not files:
            self.get_logger().warn(f"No calibration file found for stereo pair {pair_id}. Please recalibrate.")
            return False
        
        latest_file = files[0]
        cv_file = cv2.FileStorage(latest_file, cv2.FILE_STORAGE_READ)

        # Read parameters
        self.camera_matrix_left = cv_file.getNode("camera_matrix_left").mat()
        self.dist_coeffs_left = cv_file.getNode("dist_coeffs_left").mat()
        self.camera_matrix_right = cv_file.getNode("camera_matrix_right").mat()
        self.dist_coeffs_right = cv_file.getNode("dist_coeffs_right").mat()
        self.rotation_matrix = cv_file.getNode("rotation_matrix").mat()
        self.translation_vector = cv_file.getNode("translation_vector").mat()
        self.rectification_matrix_left = cv_file.getNode("rectification_matrix_left").mat()
        self.rectification_matrix_right = cv_file.getNode("rectification_matrix_right").mat()
        self.projection_matrix_left = cv_file.getNode("projection_matrix_left").mat()
        self.projection_matrix_right = cv_file.getNode("projection_matrix_right").mat()
        self.disparity_to_depth_map = cv_file.getNode("disparity_to_depth_map").mat()

        # Read rectification maps
        stereo_map_left_x = cv_file.getNode("stereo_map_left_x").mat()
        stereo_map_left_y = cv_file.getNode("stereo_map_left_y").mat()
        stereo_map_right_x = cv_file.getNode("stereo_map_right_x").mat()
        stereo_map_right_y = cv_file.getNode("stereo_map_right_y").mat()

        cv_file.release()

        # Validate calibration data
        if self.camera_matrix_left is None or self.camera_matrix_right is None:
            self.get_logger().warn(f"Calibration file {latest_file} is incomplete. Recalibration needed.")
            return False

        # Apply rectification maps
        self.rectification_maps[pair_id] = {
            "left": (stereo_map_left_x, stereo_map_left_y),
            "right": (stereo_map_right_x, stereo_map_right_y)
        }
        
        self.calibrated_pairs[pair_id] = True
        self.get_logger().info(f"Applied calibration from {latest_file}.")
        return True

    def manage_calibration_files(self, pair_id):
        """ Keeps only the last 3 calibration files, deleting the oldest if needed. """
        files = sorted(glob.glob(f"{self.calibration_dir}/stereo_calibration_params_pair_{pair_id}_*.yaml"))
        if len(files) > 3:
            oldest_file = files[0]
            os.remove(oldest_file)
            self.get_logger().info(f"Deleted oldest calibration file: {oldest_file}")

def main(args=None):
    rclpy.init(args=args)
    node = StereoCalibrationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()