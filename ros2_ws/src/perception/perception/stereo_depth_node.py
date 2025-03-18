#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import vpi
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor

class StereoDepthNode(Node):
    def __init__(self):
        super().__init__('stereo_depth_node')
        self.br = CvBridge()

        # Create subscribers for each stereo pair
        self.sub_left_0 = Subscriber(self, Image, '/camera/rectified/split_0')
        self.sub_right_0 = Subscriber(self, Image, '/camera/rectified/split_1')
        self.sub_left_1 = Subscriber(self, Image, '/camera/rectified/split_2')
        self.sub_right_1 = Subscriber(self, Image, '/camera/rectified/split_3')

        # Synchronize left & right images for each pair
        self.sync_0 = ApproximateTimeSynchronizer([self.sub_left_0, self.sub_right_0], queue_size=5, slop=0.05)
        self.sync_0.registerCallback(self.process_stereo, 0)

        self.sync_1 = ApproximateTimeSynchronizer([self.sub_left_1, self.sub_right_1], queue_size=5, slop=0.05)
        self.sync_1.registerCallback(self.process_stereo, 1)

        # Publishers for depth maps
        self.depth_publisher_0 = self.create_publisher(Image, '/camera/depth_map_0', 10)
        self.depth_publisher_1 = self.create_publisher(Image, '/camera/depth_map_1', 10)

        # Stereo Matching Parameters
        self.num_disparities = 128  # Number of disparities (must be multiple of 16)
        self.block_size = 9         # Block size for matching
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
                # quality=6,       # High quality mode
                # conftype=vpi.ConfidenceType.ABSOLUTE,
                # mindisp=0,
                # p1=3,
                # p2=48,
                # p2alpha=0,
                # uniqueness=-1.0,
                # includediagonals=True,
                # numpasses=3,
            )

            disparity_u8 = self.normalize_disparity_map(disparity_vpi, self.num_disparities)

        # Convert back to ROS Image and publish
        depth_msg = self.br.cv2_to_imgmsg(np.array(disparity_u8), encoding='mono8')
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