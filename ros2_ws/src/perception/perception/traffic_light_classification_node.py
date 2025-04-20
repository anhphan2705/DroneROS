#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs.msg import TrackedBoundingBoxes, TrackedBoundingBox
from message_filters import Subscriber, ApproximateTimeSynchronizer
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage

def classify_traffic_light_color(cropped_light):
    """Classifies traffic light color using HSV segmentation and brightness threshold."""
    try:
        cropped_light = cv2.cvtColor(cropped_light, cv2.COLOR_RGB2BGR)

        height, width = cropped_light.shape[:2]
        segment_height = height // 3
        brightness_values = []

        for i in range(3):
            segment = cropped_light[i * segment_height:(i + 1) * segment_height, 0:width]
            hsv_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
            mean_brightness = np.mean(hsv_segment[:, :, 2])
            brightness_values.append(mean_brightness)

        brightness_threshold = 100
        if max(brightness_values) < brightness_threshold:
            return 0  # OFF

        active_segment = np.argmax(brightness_values)
        if active_segment == 0:
            return 1  # RED
        elif active_segment == 1:
            return 2   # YELLOW
        else:
            return 3   # GREEN
    except Exception as e:
        print(f"Classification failed: {e}")
        return 4


class ClassificationNode(Node):
    def __init__(self):
        super().__init__('traffic_light_classification_node')

        # Parameters
        self.declare_parameter('tracked_topic', '/yolo/detections_1/depth/tracked/classified')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('classification_topic', '/yolo/detections_1/depth/tracked/classified/light')
        self.declare_parameter('class_ids_to_classify', [-1])

        self.tracked_topic = self.get_parameter('tracked_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.classification_topic = self.get_parameter('classification_topic').value
        self.class_ids_to_classify = self.get_parameter('class_ids_to_classify').value
        if self.class_ids_to_classify == [-1]:
            self.class_ids_to_classify = []

        self.bridge = CvBridge()

        # Subscribers
        img_sub = Subscriber(self, Image, self.image_topic)
        trk_sub = Subscriber(self, TrackedBoundingBoxes, self.tracked_topic)
        ats = ApproximateTimeSynchronizer([img_sub, trk_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)

        # Publisher
        self.pub = self.create_publisher(TrackedBoundingBoxes, self.classification_topic, 10)

        self.get_logger().info(f"Traffic Light Classification node is up using HSV segmentation.")

    def callback(self, img_msg: Image, trk_msg: TrackedBoundingBoxes):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return
        h, w = cv_img.shape[:2]

        out_msg = TrackedBoundingBoxes()
        out_msg.header = trk_msg.header

        for box in trk_msg.boxes:
            new_box = TrackedBoundingBox()
            new_box.id = box.id
            new_box.track_id = box.track_id
            new_box.class_id = box.class_id
            new_box.class_name = box.class_name
            new_box.confidence = box.confidence
            new_box.x_min = box.x_min
            new_box.y_min = box.y_min
            new_box.x_max = box.x_max
            new_box.y_max = box.y_max
            new_box.depth = box.depth
            new_box.speed_x = box.speed_x
            new_box.speed_y = box.speed_y
            new_box.speed_z = box.speed_z
            new_box.speed_mps = box.speed_mps

            if self.class_ids_to_classify and box.class_id not in self.class_ids_to_classify:
                new_box.classification_id = box.classification_id
            else:
                # Perform classification
                x1, y1 = int(max(0, box.x_min)), int(max(0, box.y_min))
                x2, y2 = int(min(w, box.x_max)), int(min(h, box.y_max))
                if x2 > x1 and y2 > y1:
                    roi = cv_img[y1:y2, x1:x2]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    new_box.classification_id = classify_traffic_light_color(roi_rgb)
                else:
                    new_box.classification_id = 4  # TODO: Switch to -1 after debugging
                self.get_logger().info(f"Published classified traffic light.")

            out_msg.boxes.append(new_box)
            
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ClassificationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()