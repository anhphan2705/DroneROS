#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor

class DepthVisualizerNode(Node):
    def __init__(self):
        super().__init__('depth_visualizer_node')

        self.declare_parameter('input_topic', '/camera/depth_map_0')
        self.declare_parameter('output_topic', '/camera/depth_map_0/normalized')
        self.declare_parameter('max_depth', 10.0)  # in meters

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.max_depth = self.get_parameter('max_depth').get_parameter_value().double_value

        self.bridge = CvBridge()

        self.sub = self.create_subscription(Image, self.input_topic, self.callback, 10)
        self.pub = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(f"Subscribed to: {self.input_topic}, publishing to: {self.output_topic}")

    def callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f"CVBridge error: {e}")
            return

        # Clip and normalize depth map
        depth_clipped = np.clip(depth, 0.0, self.max_depth)
        depth_normalized = (depth_clipped / self.max_depth * 255).astype(np.uint8)

        output_msg = self.bridge.cv2_to_imgmsg(depth_normalized, encoding='mono8')
        output_msg.header = msg.header
        self.pub.publish(output_msg)


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
