#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs.msg import BoundingBoxes, BoundingBox
from cv_bridge import CvBridge
import numpy as np

class ObjectDepthFusionNode(Node):
    def __init__(self):
        super().__init__('object_depth_fusion_node')

        self.declare_parameter('detection_topic', '/yolo/detections')
        self.declare_parameter('depth_topic', '/camera/depth_map_0')
        self.declare_parameter('output_topic', '/yolo/detections_with_depth')

        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.depth_frame = None

        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.bbox_sub = self.create_subscription(BoundingBoxes, self.detection_topic, self.bbox_callback, 10)
        self.pub = self.create_publisher(BoundingBoxes, self.output_topic, 10)

        self.get_logger().info("ObjectDepthFusionNode initialized.")

    def depth_callback(self, msg):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f"Depth CVBridge error: {e}")

    def bbox_callback(self, msg: BoundingBoxes):
        if self.depth_frame is None:
            self.get_logger().warn("No depth frame received yet.")
            return

        depth_boxes = BoundingBoxes()
        depth_boxes.header = msg.header

        for bbox in msg.boxes:
            x1, y1 = max(0, bbox.x_min), max(0, bbox.y_min)
            x2, y2 = min(self.depth_frame.shape[1], bbox.x_max), min(self.depth_frame.shape[0], bbox.y_max)
            roi = self.depth_frame[y1:y2, x1:x2]

            # Filter out invalid depth (0 or NaN)
            valid_depths = roi[np.isfinite(roi) & (roi > 0)]

            if valid_depths.size > 0:
                median_depth = float(np.median(valid_depths))
                bbox.depth = median_depth
            else:
                bbox.depth = -1.0  # Invalid depth

            depth_boxes.boxes.append(bbox)

        self.pub.publish(depth_boxes)
        self.get_logger().info(f"Published {len(depth_boxes.boxes)} bboxes with depth.")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDepthFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()