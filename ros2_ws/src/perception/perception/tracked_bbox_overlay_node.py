#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from msgs.msg import TrackedBoundingBoxes
from message_filters import ApproximateTimeSynchronizer, Subscriber


class TrackedBBoxOverlayNode(Node):
    def __init__(self):
        super().__init__('tracked_bbox_overlay_node')

        self.declare_parameter('image_topic', '/camera/rectified/split_0')
        self.declare_parameter('tracked_topic', '/yolo/tracked_detections')
        self.declare_parameter('output_topic', '/camera/yolo_overlay_tracked')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        tracked_topic = self.get_parameter('tracked_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.bridge = CvBridge()

        self.image_sub = Subscriber(self, Image, image_topic)
        self.tracked_sub = Subscriber(self, TrackedBoundingBoxes, tracked_topic)

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.tracked_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.callback)

        self.image_pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info("TrackedBBoxOverlayNode started — drawing track IDs on image.")

    def callback(self, img_msg: Image, tracked_msg: TrackedBoundingBoxes):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        for box in tracked_msg.boxes:
            x1, y1, x2, y2 = box.x_min, box.y_min, box.x_max, box.y_max

            label = f'Track: {box.track_id} | {box.class_name}: {box.confidence:.2f}'
            if box.depth > 0:
                label += f' ({box.depth:.2f}m)'
            if box.speed_mps > 0:
                label += f' {box.speed_mps:.2f}m/s'

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        self.get_logger().info(f"Overlayed {len(tracked_msg.boxes)} tracked boxes on image.")
        overlay_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        overlay_msg.header = img_msg.header
        self.image_pub.publish(overlay_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrackedBBoxOverlayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()