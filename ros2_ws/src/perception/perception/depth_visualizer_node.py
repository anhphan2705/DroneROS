#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor


class DepthVisualizerNode(Node):
    def __init__(self):
        super().__init__('depth_visualizer_node')

        self.declare_parameter('depth_topic', '/camera/depth_map_0')
        self.declare_parameter('output_topic', '/camera/depth_map_0/normalized')

        self.input_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, self.input_topic, self.callback, 10)
        self.pub = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(f"Subscribed to: {self.input_topic}, publishing to: {self.output_topic}")

    def callback(self, msg: Image):
        # convert to float32 array
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f"CVBridge error: {e}")
            return

        # mask out invalid depths
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        # if the entire frame is zero, skip normalization
        if not np.any(depth):
            self.get_logger().warn("Received empty depth frame, publishing zeros")
            depth_uint8 = np.zeros_like(depth, dtype=np.uint8)
        else:
            # normalize to [0,255] over full dynamic range
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_norm.astype(np.uint8)

        # convert back to ROS Image (mono8)
        out_msg = self.bridge.cv2_to_imgmsg(depth_uint8, encoding='mono8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = DepthVisualizerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()