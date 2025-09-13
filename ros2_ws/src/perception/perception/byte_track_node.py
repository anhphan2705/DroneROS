#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs.msg import BoundingBoxes, TrackedBoundingBoxes, TrackedBoundingBox
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
import math
from argparse import Namespace
import os
import cv2

class ByteTrackNode(Node):
    def __init__(self):
        super().__init__('byte_track_node')

        self.declare_parameter('input_topic', '/yolo/detections')
        self.declare_parameter('output_topic', '/yolo/tracked_detections')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('calibration_file', 'stereo_calibration_params_pair_1_2025-06-15_21-20-05.yml')

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        self.calibration_file = self.get_parameter('calibration_file').value

        self.sub = self.create_subscription(BoundingBoxes, self.input_topic, self.detection_callback, 10)
        self.pub = self.create_publisher(TrackedBoundingBoxes, self.output_topic, 10)
        
        if not os.path.isfile(self.calibration_file):
            self.get_logger().fatal(f"Calibration file not found: {self.calibration_file}")
            raise RuntimeError("Missing calibration file")

        # Parse intrinsics from YAML
        fs = cv2.FileStorage(self.calibration_file, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            self.get_logger().fatal(f"Cannot open calibration file: {self.calibration_file}")
            raise RuntimeError("Calibration file load failed")
        
        frame_size_raw = fs.getNode('frame_size').mat()
        self.image_shape = (int(frame_size_raw[1][0]), int(frame_size_raw[0][0]))  # (height, width)
        
        P1 = fs.getNode('projection_matrix_left').mat()
        self.fx = P1[0, 0]
        self.fy = P1[1, 1]
        self.cx = P1[0, 2]
        self.cy = P1[1, 2]
        fs.release()

        self.get_logger().info(
            f"Frame size: {self.image_shape}; Loaded intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}"
        )

        args = Namespace(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=30,
            frame_rate=30,
            mot20=False,
        )
        
        self.tracker = BYTETracker(args, frame_rate=30)
        self.class_map = {}

        self.get_logger().info(f"ByteTrack node subscribed to {self.input_topic}, publishing to {self.output_topic}")

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
            self.get_logger().debug("No detections — publishing empty tracked message.")
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
            tracked_box.id = int(class_id)
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
        self.get_logger().debug(f"Published {len(out_msg.boxes)} tracked boxes")

def main(args=None):
    rclpy.init(args=args)
    node = ByteTrackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()