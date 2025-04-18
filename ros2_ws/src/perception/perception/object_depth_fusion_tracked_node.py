#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs.msg import TrackedBoundingBoxes, TrackedBoundingBox
from cv_bridge import CvBridge
import numpy as np

class ObjectDepthFusionTrackedNode(Node):
    def __init__(self):
        super().__init__('object_depth_fusion_tracked_node')

        self.declare_parameter('detection_topic', '/yolo/tracked_detections')
        self.declare_parameter('depth_topic', '/camera/depth_map_0')
        self.declare_parameter('output_topic', '/yolo/tracked_detections_with_depth')

        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.depth_frame = None

        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.tracked_sub = self.create_subscription(TrackedBoundingBoxes, self.detection_topic, self.tracked_callback, 10)
        self.pub = self.create_publisher(TrackedBoundingBoxes, self.output_topic, 10)

        self.get_logger().info("ObjectDepthFusionTrackedNode initialized.")

    def depth_callback(self, msg):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f"Depth CVBridge error: {e}")

    def tracked_callback(self, msg: TrackedBoundingBoxes):
        if self.depth_frame is None:
            self.get_logger().warn("No depth frame received yet.")
            return

        out_msg = TrackedBoundingBoxes()
        out_msg.header = msg.header

        for box in msg.boxes:
            x1, y1 = max(0, box.x_min), max(0, box.y_min)
            x2, y2 = min(self.depth_frame.shape[1], box.x_max), min(self.depth_frame.shape[0], box.y_max)
            roi = self.depth_frame[y1:y2, x1:x2]

            valid_depths = roi[np.isfinite(roi) & (roi > 0)]

            if valid_depths.size > 0:
                median_depth = float(np.median(valid_depths))
                box.depth = median_depth
            else:
                box.depth = -1.0

            out_msg.boxes.append(box)

        self.pub.publish(out_msg)
        self.get_logger().info(f"Published {len(out_msg.boxes)} tracked bboxes with depth.")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDepthFusionTrackedNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
