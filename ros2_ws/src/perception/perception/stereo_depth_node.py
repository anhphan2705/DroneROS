#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

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
        self.sync_0 = ApproximateTimeSynchronizer([self.sub_left_0, self.sub_right_0], queue_size=10, slop=0.1)
        self.sync_0.registerCallback(self.process_stereo, 0)

        self.sync_1 = ApproximateTimeSynchronizer([self.sub_left_1, self.sub_right_1], queue_size=10, slop=0.1)
        self.sync_1.registerCallback(self.process_stereo, 1)

        # Publishers for depth maps
        self.depth_publisher_0 = self.create_publisher(Image, '/camera/depth_map_0', 10)
        self.depth_publisher_1 = self.create_publisher(Image, '/camera/depth_map_1', 10)

        # Stereo Matching Setup (Stereo SGBM)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,  # Adjust based on scene depth
            blockSize=9,
            P1=8 * 3 * 9**2,
            P2=32 * 3 * 9**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=2
        )

        self.get_logger().info("StereoDepthNode Initialized")

    def process_stereo(self, left_msg, right_msg, pair_id):
        """Process left and right images to generate depth map."""
        try:
            left_image = self.br.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            right_image = self.br.imgmsg_to_cv2(right_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
        
        # Compute disparity map
        disparity = self.stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        
        # Normalize for visualization
        disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_visual = disparity_visual.astype(np.uint8)

        # Convert back to ROS Image and publish
        depth_msg = self.br.cv2_to_imgmsg(disparity_visual, encoding='mono8')
        depth_msg.header.stamp = left_msg.header.stamp  # Keep timestamp for sync

        if pair_id == 0:
            self.depth_publisher_0.publish(depth_msg)
        else:
            self.depth_publisher_1.publish(depth_msg)

        # self.get_logger().info(f"Published depth map for pair {pair_id}")

def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()