#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import vpi
import os
import glob
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory

class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')
        self.br = CvBridge()
        
        self.declare_parameter('sub_left', '/camera/rectified/split_0')
        self.declare_parameter('sub_right', '/camera/rectified/split_1')
        self.declare_parameter('depth_publisher', '/camera/depth_map_0')
        self.declare_parameter('calibration_file', 'stereo_calibration_params_pair_0_2025-04-28_16-39-56.yml')
        
        self.sub_left = self.get_parameter('sub_left').get_parameter_value().string_value
        self.sub_right = self.get_parameter('sub_right').get_parameter_value().string_value
        self.depth_publisher = self.get_parameter('depth_publisher').get_parameter_value().string_value
        calib_file = self.get_parameter('calibration_file').get_parameter_value().string_value

        if not calib_file:
            # fallback: pick latest in calibrated_params folder
            calib_dir = os.path.join(
                get_package_share_directory('calibration'),
                'calibrated_params'
            )
            matches = sorted(glob.glob(os.path.join(calib_dir,
                        'stereo_calibration_params_pair_0_*.yml')), reverse=True)
            if not matches:
                self.get_logger().fatal("No calibration file found!")
                raise RuntimeError("Calibration missing")
            calib_file = matches[0]

        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            self.get_logger().fatal(f"Unable to open {calib_file}")
            raise RuntimeError("Cannot read calibration")

        # Intrinsics
        M1 = fs.getNode('camera_matrix_left').mat()
        # D1 = fs.getNode('dist_coeffs_left').mat()
        # M2 = fs.getNode('camera_matrix_right').mat()
        # D2 = fs.getNode('dist_coeffs_right').mat()
        # Extrinsics between cameras
        # R  = fs.getNode('rotation_matrix').mat()
        T  = fs.getNode('translation_vector').mat()   # in same units as square_size (mm)
        # # Rectification & projection
        # R1 = fs.getNode('rectification_matrix_left').mat()
        # R2 = fs.getNode('rectification_matrix_right').mat()
        # P1 = fs.getNode('projection_matrix_left').mat()
        # P2 = fs.getNode('projection_matrix_right').mat()
        # (optional) disparity→depth map
        # Q  = fs.getNode('disparity_to_depth_map').mat()
        fs.release()

        # Compute fx & baseline from P1/P2 or T
        self.fx = M1[0,0]
        # translation_vector is in mm; convert to meters and abs in case of sign
        self.baseline_m = abs(T[0,0]) / 1000.0
        
        # Create subscribers for each stereo pair
        self.sub_left = Subscriber(self, Image, self.sub_left)
        self.sub_right = Subscriber(self, Image, self.sub_right)

        # Synchronize left & right images for each pair
        self.sync = ApproximateTimeSynchronizer([self.sub_left, self.sub_right], queue_size=5, slop=0.05)
        self.sync.registerCallback(self.process_stereo)

        # Publishers for depth maps
        self.depth_publisher = self.create_publisher(Image, self.depth_publisher, 10)

        # Stereo Matching Parameters
        self.num_disparities = 128  # Number of disparities (must be multiple of 16)
        self.block_size = 3         # Block size for matching (1, 3, 5, or 7)
        # self.num_pass = 1
        # self.quality = 1
        self.uniqueness = -1.0
        self.includediagonals = False
        self.stream = vpi.Stream()  # Create a reusable VPI stream

        self.get_logger().info(f"StereoDepthNode Initialized with calibration file {calib_file}!")

    def normalize_disparity_map(self, disparity_S16, max_disparity):
        """Normalize disparity map for visualization."""
        return disparity_S16.convert(vpi.Format.U8, scale=255.0 / (32 * max_disparity)).cpu()

    def process_stereo(self, left_msg, right_msg):
        """Process left and right images to generate depth map using VPI."""
        try:
            left_cv = self.br.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            right_cv = self.br.imgmsg_to_cv2(right_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
        
        # Convert images to VPI format with CUDA stream
        with self.stream, vpi.Backend.CUDA:
            left_vpi = vpi.asimage(left_cv).convert(vpi.Format.Y16_ER)
            right_vpi = vpi.asimage(right_cv).convert(vpi.Format.Y16_ER)

        # Stereo Disparity Estimation with VPI
        with self.stream, vpi.Backend.CUDA:
            disparity_vpi = vpi.stereodisp(
                left=left_vpi,
                right=right_vpi,
                backend=vpi.Backend.CUDA,
                maxdisp=self.num_disparities,      # Maximum disparity
                window=self.block_size,        # Median filter window size
                # confthreshold=32767,
                # quality=self.quality,
                # conftype=vpi.ConfidenceType.ABSOLUTE,
                mindisp=0,
                # p1=3,
                # p2=48,
                # p2alpha=0,
                uniqueness=self.uniqueness,
                includediagonals=self.includediagonals,
                # numpasses=self.num_pass,
            )

            disparity_float = disparity_vpi.convert(vpi.Format.F32).cpu() / 32.0

        # self.get_logger().info(
        #     f"Disparity stats: min={np.min(disparity_float):.2f}, max={np.max(disparity_float):.2f}, mean={np.mean(disparity_float):.2f}"
        # )

        # Allocate depth map array with the same shape as the disparity image
        valid = disparity_float > 0
        depth_map = np.zeros_like(disparity_float, dtype=np.float32)
        depth_map[valid] = (self.fx * self.baseline_m) / disparity_float[valid]
        
        # if np.any(valid):
        #     self.get_logger().info(
        #         f"Depth stats: min={np.min(depth_map[valid]):.2f}, max={np.max(depth_map[valid]):.2f}, mean={np.mean(depth_map[valid]):.2f}"
        #     )

        # Convert the computed depth map into a ROS Image message.
        depth_msg = self.br.cv2_to_imgmsg(depth_map, encoding='32FC1')
        depth_msg.header = left_msg.header

        self.depth_publisher.publish(depth_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()

    # Use MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()