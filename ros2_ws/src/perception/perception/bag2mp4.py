#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np

class Bag2Mp4(Node):
    def __init__(self):
        super().__init__('bag2mp4')
        self.subscription = self.create_subscription(
            Image,
            '/camera/rectified_0/depth_map',
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        self.out = None
        self.frame_count = 0
        self.fps = 30
        self.video_path = 'output.mp4'
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')

        self.last_frame_time = self.get_clock().now()
        self.timeout_seconds = 5.0
        self.timer = self.create_timer(1.0, self.check_timeout)
        
    def check_valid_frame(self, image_frame):
        self.get_logger().info(f"Frame size: {image_frame.shape}")
        self.get_logger().info(f"Writer is open: {self.out.isOpened()}")
        
    # Regular Image topic conversion
    # def listener_callback(self, msg):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    #         if self.out is None:
    #             h, w = cv_image.shape[:2]
    #             self.out = cv2.VideoWriter(self.video_path, self.codec, self.fps, (w, h))
    #             self.get_logger().info(f"Recording started: {self.video_path}")
    #             self.check_valid_frame(cv_image)

    #         self.get_logger().info(f"Wrote frame {self.frame_count}")
    #         self.out.write(cv_image)
    #         self.frame_count += 1
    #         self.last_frame_time = self.get_clock().now()

    #     except Exception as e:
    #         self.get_logger().error(f"Failed to process frame: {e}")

    # 32FC1 Depth map to Mp4
    def listener_callback(self, msg):
        try:
            # Convert depth image (32FC1) to NumPy array
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Normalize depth to 0–255 and convert to uint8
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = np.uint8(depth_normalized)

            # Apply color map
            depth_bgr = cv2.cvtColor(depth_uint8, cv2.COLOR_GRAY2BGR)

            if self.out is None:
                h, w = depth_bgr.shape[:2]
                self.out = cv2.VideoWriter(self.video_path, self.codec, self.fps, (w, h))
                self.get_logger().info(f"Recording started: {self.video_path}")
                self.check_valid_frame(depth_bgr)
    
            self.get_logger().info(f"Wrote frame {self.frame_count}")
            self.out.write(depth_bgr)
            self.frame_count += 1
            self.last_frame_time = self.get_clock().now()

        except Exception as e:
            self.get_logger().error(f"Failed to process depth frame: {e}")
            
    def check_timeout(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_frame_time).nanoseconds / 1e9
        if elapsed > self.timeout_seconds:
            self.out.release()
            self.get_logger().info(f"No frames for {elapsed:.1f} sec — stopping recording.")
            rclpy.shutdown()

    def destroy_node(self):
        if self.out:
            self.out.release()
            if self.frame_count == 0 and os.path.exists(self.video_path):
                os.remove(self.video_path)
                self.get_logger().info(f"No frames recorded — deleted {self.video_path}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    recorder = Bag2Mp4()
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    recorder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()