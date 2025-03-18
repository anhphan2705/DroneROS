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
        self.num_disparities = 64  # Number of disparities (must be multiple of 16)
        self.block_size = 9        # Block size for matching

        self.get_logger().info("StereoDepthNode Initialized!")

    def process_stereo(self, left_msg, right_msg, pair_id):
        """Process left and right images to generate a disparity map using VPI."""
        try:
            left_image = self.br.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            right_image = self.br.imgmsg_to_cv2(right_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        height, width = left_image.shape

        # Convert to VPI Images
        with vpi.Backend.CUDA:
            vpi_left = vpi.asimage(left_image, vpi.Format.U8)
            vpi_right = vpi.asimage(right_image, vpi.Format.U8)

            # Ensure disparity image format is S16
            disparity_vpi = vpi.Image((width, height), vpi.Format.S16)

            # Run stereo disparity estimation
            with vpi.Backend.CUDA:
                with vpi.Stream() as stream:
                    vpi.stereodisp(left=vpi_left, right=vpi_right, 
                                maxdisp=self.num_disparities,
                                window=self.block_size,
                                backend=vpi.Backend.CUDA, 
                                out=disparity_vpi)

            # Convert disparity back to NumPy
            disparity_np = disparity_vpi.cpu().astype(np.float32) / 16.0
            
        # Normalize for visualization
        disparity_visual = cv2.normalize(disparity_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_visual = disparity_visual.astype(np.uint8)

        # Convert back to ROS Image and publish
        depth_msg = self.br.cv2_to_imgmsg(disparity_visual, encoding='mono8')
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