#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import math
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
import os

class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__('depth2pcl_node')
        self.br = CvBridge()

        # Declare parameters
        self.declare_parameter('depth_topic', '/camera/depth_map_1')
        self.declare_parameter('pointcloud_topic', '/camera/depth_cloud_1')
        self.declare_parameter('calibration_file', 'stereo_calibration_params_pair_1_2025-06-15_21-20-05.yml')
        self.declare_parameter('max_depth', 40.0)

        self.depth_topic = self.get_parameter('depth_topic').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.calibration_file = self.get_parameter('calibration_file').value
        self.max_depth = self.get_parameter('max_depth').value

        if not os.path.isfile(self.calibration_file):
            self.get_logger().fatal(f"Calibration file not found: {self.calibration_file}")
            raise RuntimeError("Missing calibration file")

        # Parse intrinsics from YAML
        fs = cv2.FileStorage(self.calibration_file, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            self.get_logger().fatal(f"Cannot open calibration file: {self.calibration_file}")
            raise RuntimeError("Calibration file load failed")

        P1 = fs.getNode('projection_matrix_left').mat()
        self.fx = P1[0, 0]
        self.fy = P1[1, 1]
        self.cx = P1[0, 2]
        self.cy = P1[1, 2]
        fs.release()

        self.get_logger().info(
            f"Loaded intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
        )

        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.pcl_pub = self.create_publisher(PointCloud2, self.pointcloud_topic, 10)

    def depth_callback(self, msg):
        try:
            depth_img = self.br.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            depth_img = cv2.medianBlur(depth_img, 5)
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")
            return
        
        depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
        depth_img[depth_img > self.max_depth] = 0.0
        height, width = depth_img.shape
        points = []

        for v in range(0, height, 6):
            for u in range(0, width, 6):
                Z = float(depth_img[v, u])
                if Z <= 0.1 or Z > self.max_depth or math.isnan(Z):
                    points.append((float('nan'), float('nan'), float('nan')))
                    continue

                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy
                points.append((X, Y, Z))

        if points:
            cloud_msg = point_cloud2.create_cloud_xyz32(
                msg.header,
                points
            )
            cloud_msg.height = 1
            cloud_msg.width = len(points)
            cloud_msg.is_dense = False
            cloud_msg.header.frame_id = msg.header.frame_id or 'camera_link'
            self.pcl_pub.publish(cloud_msg)
        else:
            self.get_logger().warn("No valid depth points found in frame.")

def main(args=None):
    rclpy.init(args=args)
    node = DepthToPointCloud()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
