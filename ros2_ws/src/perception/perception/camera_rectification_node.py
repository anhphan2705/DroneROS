#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.executors import MultiThreadedExecutor

import glob, os
import numpy as np
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from functools import partial
from ament_index_python.packages import get_package_share_directory

class RectificationNode(Node):
    def __init__(self):
        super().__init__('camera_rectification_node')
        self.br = CvBridge()

        pkg_dir = get_package_share_directory('perception')
        self.calib_dir = os.path.join(pkg_dir, 'calibrated_params')

        # maps[pair_id] = {'left': (mapX,mapY), 'right': (...)}
        self.maps = {}
        self.topic_to_pair = {
            '/camera/image_raw/split_0': (0, 'left'),
            '/camera/image_raw/split_1': (0, 'right'),
            '/camera/image_raw/split_2': (1, 'left'),
            '/camera/image_raw/split_3': (1, 'right'),
        }

        # load both stereo pairs
        ok0 = self._load_maps(0)
        ok1 = self._load_maps(1)
        if not (ok0 and ok1):
            self.get_logger().error('Failed to load calibration for one or more pairs')
            rclpy.shutdown()
            return

        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # subscribe & publisher dict
        self.pubs = {}
        for topic, (pid, side) in self.topic_to_pair.items():
            self.create_subscription(
                Image, topic,
                partial(self.image_cb, topic=topic),
                qos_profile=qos
            )
            rect_topic = f'/camera/rectified/{topic.split("/")[-1]}'
            self.pubs[topic] = self.create_publisher(Image, rect_topic, qos_profile=qos)
            self.get_logger().info(f'Rectify: {topic} → {rect_topic}')

    def _load_maps(self, pair_id: int) -> bool:
        """Load stereo_map_* arrays from latest YAML for given pair."""
        pattern = os.path.join(
            self.calib_dir,
            f'stereo_calibration_params_pair_{pair_id}_*.yml'
        )
        files = sorted(glob.glob(pattern), reverse=True)
        if not files:
            self.get_logger().warn(f'No calib file for pair {pair_id}')
            return False

        fn = files[0]
        self.get_logger().info(f'Loading maps from {fn}')
        fs = cv2.FileStorage(fn, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            self.get_logger().error(f'Cannot open {fn}')
            return False

        # read directly
        mapLx = fs.getNode('stereo_map_left_x').mat()
        mapLy = fs.getNode('stereo_map_left_y').mat()
        mapRx = fs.getNode('stereo_map_right_x').mat()
        mapRy = fs.getNode('stereo_map_right_y').mat()
        fs.release()

        if any(m is None for m in [mapLx, mapLy, mapRx, mapRy]):
            self.get_logger().error(f'Maps missing in {fn}')
            return False

        self.maps[pair_id] = {
            'left':  (mapLx, mapLy),
            'right': (mapRx, mapRy),
        }
        return True

    def image_cb(self, msg: Image, topic: str):
        pid, side = self.topic_to_pair[topic]
        mapX, mapY = self.maps[pid][side]

        try:
            cv_img = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        # single cpu remap
        rect = cv2.remap(cv_img, mapX, mapY, interpolation=cv2.INTER_LINEAR)

        # back to ROS Image
        out = self.br.cv2_to_imgmsg(rect, encoding='bgr8')
        out.header = msg.header
        self.pubs[topic].publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = RectificationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
