#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import vpi
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor

class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')
        self.br = CvBridge()

        self.declare_parameter('sub_left_0', '/camera/rectified/split_0')
        self.declare_parameter('sub_right_0', '/camera/rectified/split_1')
        self.declare_parameter('sub_left_1', '/camera/rectified/split_2')
        self.declare_parameter('sub_right_1', '/camera/rectified/split_3')
        self.declare_parameter('depth_publisher_0', '/camera/depth_map_0')
        self.declare_parameter('depth_publisher_1', '/camera/depth_map_1')
        self.declare_parameter('horizontal_fov_deg', 66.0)
        self.declare_parameter('baseline_m', 0.05)  # 50 mm
        
        self.sub_left_0 = self.get_parameter('sub_left_0').get_parameter_value().string_value
        self.sub_right_0 = self.get_parameter('sub_right_0').get_parameter_value().string_value
        self.sub_left_1 = self.get_parameter('sub_left_1').get_parameter_value().string_value
        self.sub_right_1 = self.get_parameter('sub_right_1').get_parameter_value().string_value
        self.depth_publisher_0 = self.get_parameter('depth_publisher_0').get_parameter_value().string_value
        self.depth_publisher_1 = self.get_parameter('depth_publisher_1').get_parameter_value().string_value
        self.horizontal_fov_deg = self.get_parameter('horizontal_fov_deg').value
        self.baseline_m = self.get_parameter('baseline_m').value        
        
        # Create subscribers for each stereo pair
        self.sub_left_0 = Subscriber(self, Image, self.sub_left_0)
        self.sub_right_0 = Subscriber(self, Image, self.sub_right_0)
        self.sub_left_1 = Subscriber(self, Image, self.sub_left_1)
        self.sub_right_1 = Subscriber(self, Image, self.sub_right_1)


        # Synchronize left & right images for each pair
        self.sync_0 = ApproximateTimeSynchronizer([self.sub_left_0, self.sub_right_0], queue_size=5, slop=0.05)
        self.sync_0.registerCallback(self.process_stereo, 0)

        self.sync_1 = ApproximateTimeSynchronizer([self.sub_left_1, self.sub_right_1], queue_size=5, slop=0.05)
        self.sync_1.registerCallback(self.process_stereo, 1)

        # Publishers for depth maps
        self.depth_publisher_0 = self.create_publisher(Image, self.depth_publisher_0, 10)
        self.depth_publisher_1 = self.create_publisher(Image, self.depth_publisher_1, 10)

        # Stereo Matching Parameters
        self.num_disparities = 64  # Number of disparities (must be multiple of 16)
        self.block_size = 3         # Block size for matching (1, 3, 5, or 7)
        # self.num_pass = 1
        # self.quality = 1
        self.uniqueness = -1.0
        self.includediagonals = False
        self.stream = vpi.Stream()  # Create a reusable VPI stream

        self.get_logger().info("StereoDepthNode Initialized!")

    def normalize_disparity_map(self, disparity_S16, max_disparity):
        """Normalize disparity map for visualization."""
        return disparity_S16.convert(vpi.Format.U8, scale=255.0 / (32 * max_disparity)).cpu()

    def process_stereo(self, left_msg, right_msg, pair_id):
        """Process left and right images to generate depth map using VPI."""
        try:
            left_cv = self.br.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            right_cv = self.br.imgmsg_to_cv2(right_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # Compute effective focal length in pixels from image width and FOV:
        image_width = left_cv.shape[1]  # assuming left and right images have same width
        focal_length_px = (image_width / 2) / math.tan(math.radians(self.horizontal_fov_deg / 2))
        
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
                # mindisp=0,
                # p1=3,
                # p2=48,
                # p2alpha=0,
                uniqueness=self.uniqueness,
                includediagonals=self.includediagonals,
                # numpasses=self.num_pass,
            )

            disparity_float = disparity_vpi.convert(vpi.Format.F32).cpu() / 64

        # self.get_logger().info(
        #     f"Disparity stats: min={np.min(disparity_float):.2f}, max={np.max(disparity_float):.2f}, mean={np.mean(disparity_float):.2f}"
        # )

        # Allocate depth map array with the same shape as the disparity image
        valid = disparity_float > 0
        depth_map = np.zeros_like(disparity_float, dtype=np.float32)
        depth_map[valid] = (focal_length_px * self.baseline_m) / disparity_float[valid]
        
        # if np.any(valid):
        #     self.get_logger().info(
        #         f"Depth stats: min={np.min(depth_map[valid]):.2f}, max={np.max(depth_map[valid]):.2f}, mean={np.mean(depth_map[valid]):.2f}"
        #     )

        # Convert the computed depth map into a ROS Image message.
        depth_msg = self.br.cv2_to_imgmsg(depth_map, encoding='32FC1')
        depth_msg.header = left_msg.header

        if pair_id == 0:
            self.depth_publisher_0.publish(depth_msg)
        else:
            self.depth_publisher_1.publish(depth_msg)

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