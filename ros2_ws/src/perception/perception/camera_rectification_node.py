#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import cv2, os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def load_rectification_maps(calib_file: str):
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration file {calib_file}")
    mapLx = fs.getNode('stereo_map_left_x').mat()
    mapLy = fs.getNode('stereo_map_left_y').mat()
    mapRx = fs.getNode('stereo_map_right_x').mat()
    mapRy = fs.getNode('stereo_map_right_y').mat()
    fs.release()
    if any(m is None for m in (mapLx, mapLy, mapRx, mapRy)):
        raise RuntimeError(f"Missing rectification maps in {calib_file}")
    return {'left': (mapLx, mapLy), 'right': (mapRx, mapRy)}

class RectificationNode(Node):
    def __init__(self):
        super().__init__('rectification_node')
        # Declare launch parameters
        self.declare_parameter('left_image_topic', '/camera/image_raw/left')
        self.declare_parameter('right_image_topic', '/camera/image_raw/right')
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('output_prefix', '/camera/rectified')

        left_topic = self.get_parameter('left_image_topic').value
        right_topic = self.get_parameter('right_image_topic').value
        calib_file = self.get_parameter('calibration_file').value
        out_prefix = self.get_parameter('output_prefix').value.rstrip('/')

        if not calib_file or not os.path.isfile(calib_file):
            self.get_logger().error(f"Invalid calibration_file: {calib_file}")
            rclpy.shutdown()
            return

        # Load rectification maps
        try:
            self.maps = load_rectification_maps(calib_file)
        except RuntimeError as e:
            self.get_logger().error(str(e))
            rclpy.shutdown()
            return

        # QoS settings
        qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE
        )

        # Publishers
        left_out = f"{out_prefix}/left"
        right_out = f"{out_prefix}/right"
        self.left_pub = self.create_publisher(Image, left_out, qos)
        self.right_pub = self.create_publisher(Image, right_out, qos)

        # Subscribers
        self.br = CvBridge()
        self.create_subscription(
            Image, left_topic,
            lambda msg: self.image_cb(msg, 'left'),
            qos_profile=qos
        )
        self.create_subscription(
            Image, right_topic,
            lambda msg: self.image_cb(msg, 'right'),
            qos_profile=qos
        )

        self.get_logger().info(
            f"Rectifying:\n  {left_topic} → {left_out}\n  {right_topic} → {right_out}"
        )

    def image_cb(self, msg: Image, side: str):
        mapX, mapY = self.maps[side]
        try:
            cv_img = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return
        rect = cv2.remap(cv_img, mapX, mapY, interpolation=cv2.INTER_LINEAR)
        out = self.br.cv2_to_imgmsg(rect, encoding='bgr8')
        out.header = msg.header
        if side == 'left':
            self.left_pub.publish(out)
        else:
            self.right_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RectificationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()