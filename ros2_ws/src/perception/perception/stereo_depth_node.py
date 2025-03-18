#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
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

        # # **Use StereoBM for faster computation**
        # self.stereo = cv2.StereoBM_create(
        #     numDisparities=32,   # Reduce disparity range for speed
        #     blockSize=7          # Reduce block size for faster computation
        # )

        # If CUDA is available, use GPU acceleration
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.get_logger().info("CUDA is available, using GPU acceleration")
            # self.stereo = cv2.cuda.StereoBM_create(
            #     numDisparities=32,
            #     blockSize=7
            # )
            self.stereo = cv2.cuda.StereoSGBM_create(
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
            self.use_cuda = True
        else:
            self.get_logger().info("CUDA is not available, using CPU")
            self.use_cuda = False

        self.get_logger().info("StereoDepthNode Initialized and Running")

    def process_stereo(self, left_msg, right_msg, pair_id):
        """Process left and right images to generate depth map."""
        try:
            left_image = self.br.imgmsg_to_cv2(left_msg, desired_encoding='mono8')
            right_image = self.br.imgmsg_to_cv2(right_msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return
        
        # # **Resize images for faster computation**
        # left_image = cv2.resize(left_image, (480, 270))  # Half resolution
        # right_image = cv2.resize(right_image, (480, 270))

        # Compute disparity map
        if self.use_cuda:
            left_gpu = cv2.cuda_GpuMat()
            right_gpu = cv2.cuda_GpuMat()
            left_gpu.upload(left_image)
            right_gpu.upload(right_image)
            disparity = self.stereo.compute(left_gpu, right_gpu).download().astype(np.float32) / 16.0
        else:
            disparity = self.stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        
        # Normalize for visualization
        disparity_visual = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
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