#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import datetime
from msgs.srv import CaptureImageRequest

class SimpleCaptureNode(Node):
    def __init__(self):
        super().__init__('simple_capture_node')

        # Declare parameter for image topic
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.topic = self.get_parameter("camera_topic").get_parameter_value().string_value

        self.bridge = CvBridge()

        # Folder to save images
        self.save_directory = 'image_captured'
        os.makedirs(self.save_directory, exist_ok=True)

        # Subscribe to camera topic
        self.sub = self.create_subscription(Image, self.topic, self.image_callback, 10)
        self.last_image = None

        # Service to trigger capture
        self.srv = self.create_service(CaptureImageRequest,
                                       'capture_image',
                                       self.service_callback)

        self.get_logger().info(f"Ready to capture from {self.topic}")

    def image_callback(self, msg: Image):
        """Keep the latest image in memory."""
        self.last_image = msg

    def service_callback(self, request, response):
        """Save the latest image when service is called."""
        if self.last_image is None:
            response.success = False
            response.message = "No image received yet."
            return response

        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.last_image, desired_encoding='bgr8')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_directory, f"capture_{timestamp}.png")
            cv2.imwrite(filename, cv_image)
            response.success = True
            response.message = f"Saved image to {filename}"
            self.get_logger().info(response.message)
        except CvBridgeError as e:
            response.success = False
            response.message = f"Conversion error: {e}"
            self.get_logger().error(response.message)

        return response


def main(args=None):
    rclpy.init(args=args)
    node = SimpleCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


# How to use it: ros2 service call /capture_image msgs/srv/CaptureImageRequest "{}"