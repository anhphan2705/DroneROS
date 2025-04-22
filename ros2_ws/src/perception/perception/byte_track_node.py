#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs.msg import BoundingBoxes, TrackedBoundingBoxes, TrackedBoundingBox
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
import math
from argparse import Namespace

class ByteTrackNode(Node):
    def __init__(self):
        super().__init__('byte_track_node')

        self.declare_parameter('input_topic', '/yolo/detections')
        self.declare_parameter('output_topic', '/yolo/tracked_detections')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('horizontal_fov_deg', 66.0)
        self.declare_parameter('vertical_fov_deg', 49.5)

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.horizontal_fov_deg = self.get_parameter('horizontal_fov_deg').value
        self.vertical_fov_deg = self.get_parameter('vertical_fov_deg').value

        self.sub = self.create_subscription(BoundingBoxes, self.input_topic, self.detection_callback, 10)
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 1)
        self.pub = self.create_publisher(TrackedBoundingBoxes, self.output_topic, 10)

        args = Namespace(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=30,
            frame_rate=30,
            mot20=False,
        )
        self.tracker = BYTETracker(args, frame_rate=30)

        self.fx = None
        self.fy = None
        self.image_shape = None
        self.focal_length_computed = False
        self.class_map = {}

        self.get_logger().info(f"ByteTrack node subscribed to {self.input_topic}, publishing to {self.output_topic}")
        
    def image_callback(self, msg: Image):
        if self.focal_length_computed:
            return

        height = msg.height
        width = msg.width
        self.image_shape = (height, width)
        aspect_ratio = width / height

        # Compute fx, fy from known FOVs
        if self.fx is None and self.horizontal_fov_deg > 0:
            self.fx = (width / 2) / math.tan(math.radians(self.horizontal_fov_deg / 2))
        if self.fy is None and self.vertical_fov_deg > 0:
            self.fy = (height / 2) / math.tan(math.radians(self.vertical_fov_deg / 2))

        # Infer vertical FOV from horizontal FOV if vertical is missing
        if self.vertical_fov_deg < 0 and self.horizontal_fov_deg > 0:
            self.vertical_fov_deg = math.degrees(2 * math.atan(math.tan(math.radians(self.horizontal_fov_deg / 2)) / aspect_ratio))
            self.fy = (height / 2) / math.tan(math.radians(self.vertical_fov_deg / 2))
            self.get_logger().warn(f"Inferred vertical FOV: {self.vertical_fov_deg:.2f}°")

        # Infer horizontal FOV from vertical FOV if horizontal is missing
        if self.horizontal_fov_deg < 0 and self.vertical_fov_deg > 0:
            self.horizontal_fov_deg = math.degrees(2 * math.atan(math.tan(math.radians(self.vertical_fov_deg / 2)) * aspect_ratio))
            self.fx = (width / 2) / math.tan(math.radians(self.horizontal_fov_deg / 2))
            self.get_logger().warn(f"Inferred horizontal FOV: {self.horizontal_fov_deg:.2f}°")

        if self.fx is None or self.fy is None:
            self.get_logger().error("Failed to compute fx/fy due to missing FOV values.")
            return

        self.get_logger().info(f"Image shape: {self.image_shape}, fx: {self.fx:.2f}, fy: {self.fy:.2f}")
        self.focal_length_computed = True

    def detection_callback(self, msg: BoundingBoxes):
        
        if self.image_shape is None:
            self.get_logger().warn("Waiting for image to determine image_shape")
            return
        
        detections = []

        if self.fx is None or self.image_shape is None:
            self.get_logger().warn("Inaccurate speed estimation: fx/fy not calculated")
        
        for box in msg.boxes:
            x1, y1, x2, y2 = box.x_min, box.y_min, box.x_max, box.y_max
            conf = box.confidence
            class_name = box.class_name
            class_id = box.class_id

            detections.append([x1, y1, x2, y2, conf, class_id])
            
            if class_id not in self.class_map and class_name:
                self.class_map[class_id] = class_name

            self.image_shape = (
                max(self.image_shape[0], y2),
                max(self.image_shape[1], x2)
            )

        img_info = self.image_shape
        img_size = self.image_shape

        if len(detections) == 0:
            self.get_logger().info("No detections — publishing empty tracked message.")
            empty_msg = TrackedBoundingBoxes()
            empty_msg.header = msg.header
            self.pub.publish(empty_msg)
            return

        detections_np = np.array(detections, dtype=np.float32)
        outputs = self.tracker.update(detections_np, img_info, img_size)

        out_msg = TrackedBoundingBoxes()
        out_msg.header = msg.header

        for track in outputs:
            tlbr = track.tlbr
            track_id = track.track_id
            class_id = int(track.class_id)
            class_name = self.class_map.get(class_id, "unknown")

            tracked_box = TrackedBoundingBox()
            tracked_box.id = int(track_id)
            tracked_box.track_id = int(track_id)
            tracked_box.class_id = int(class_id)
            tracked_box.classification_id = int(-1)
            tracked_box.class_name = str(class_name)
            tracked_box.confidence = float(track.score)
            tracked_box.x_min = int(tlbr[0])
            tracked_box.y_min = int(tlbr[1])
            tracked_box.x_max = int(tlbr[2])
            tracked_box.y_max = int(tlbr[3])
            tracked_box.depth = -1.0
            tracked_box.speed_x = -1.0
            tracked_box.speed_y = -1.0
            tracked_box.speed_z = -1.0
            tracked_box.speed_mps = -1.0

            out_msg.boxes.append(tracked_box)

        self.pub.publish(out_msg)
        self.get_logger().info(f"Published {len(out_msg.boxes)} tracked boxes")

def main(args=None):
    rclpy.init(args=args)
    node = ByteTrackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()