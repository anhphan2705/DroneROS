#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from msgs.msg import TrackedBoundingBoxes, TrackedBoundingBox
from message_filters import Subscriber, ApproximateTimeSynchronizer
from collections import deque

MAX_HISTORY = 7
FLASH_THRESHOLD = 10

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
            return 0, brightness_values  # OFF

        active_segment = np.argmax(brightness_values)
        return [1, 2, 3][active_segment], brightness_values  # RED, YELLOW, GREEN
    except Exception as e:
        print(f"Classification failed: {e}")
        return -1, [0, 0, 0]  # UNKNOWN


class TrafficLightClassification(Node):
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
        self.flash_history = {}  # track_id -> deque of brightness vectors

        img_sub = Subscriber(self, Image, self.image_topic)
        trk_sub = Subscriber(self, TrackedBoundingBoxes, self.tracked_topic)
        ats = ApproximateTimeSynchronizer([img_sub, trk_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)

        self.pub = self.create_publisher(TrackedBoundingBoxes, self.classification_topic, 10)
        self.get_logger().info("Traffic Light Classification node is up using HSV segmentation.")

    def callback(self, img_msg: Image, tracked_msg: TrackedBoundingBoxes):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        h, w = cv_img.shape[:2]
        updated_boxes = []

        for box in tracked_msg.boxes:
            if self.class_ids_to_classify and box.class_id not in self.class_ids_to_classify:
                updated_boxes.append(box)
                continue

            x1 = max(0, box.x_min)
            y1 = max(0, box.y_min)
            x2 = min(w, box.x_max)
            y2 = min(h, box.y_max)

            if x2 <= x1 or y2 <= y1:
                box.classification_id = 9  # Invalid ROI
                updated_boxes.append(box)
                continue

            roi = cv_img[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            result, brightness_vector = classify_traffic_light_color(roi_rgb)

            if box.track_id not in self.flash_history:
                self.flash_history[box.track_id] = deque(maxlen=MAX_HISTORY)
            self.flash_history[box.track_id].append(brightness_vector)

            if len(self.flash_history[box.track_id]) >= MAX_HISTORY:
                brightness_array = np.array(self.flash_history[box.track_id])
                active_segment = np.argmax(np.mean(brightness_array, axis=0))
                variation = np.std(brightness_array[:, active_segment])

                if variation > FLASH_THRESHOLD:
                    if active_segment == 0:
                        result = 9  # FLASHING RED
                    elif active_segment == 1:
                        result = 8  # FLASHING YELLOW

            box.classification_id = result
            self.get_logger().info(f"Traffic light classified to ID: {result}")
            updated_boxes.append(box)

        out_msg = TrackedBoundingBoxes()
        out_msg.header = tracked_msg.header
        out_msg.boxes = updated_boxes
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightClassification()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()