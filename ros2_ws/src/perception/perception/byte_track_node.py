#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from msgs.msg import BoundingBoxes, TrackedBoundingBoxes, TrackedBoundingBox
from yolox.tracker.byte_tracker import BYTETracker
import numpy as np
from argparse import Namespace

class ByteTrackNode(Node):
    def __init__(self):
        super().__init__('byte_track_node')

        self.declare_parameter('input_topic', '/yolo/detections')
        self.declare_parameter('output_topic', '/yolo/tracked_detections')

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value

        self.sub = self.create_subscription(BoundingBoxes, self.input_topic, self.detection_callback, 10)
        self.pub = self.create_publisher(TrackedBoundingBoxes, self.output_topic, 10)

        # FIX: Use Namespace for args as required by BYTETracker
        args = Namespace(
            track_thresh=0.5,
            match_thresh=0.8,
            track_buffer=30,
            frame_rate=30,
            mot20=False,
        )
        self.tracker = BYTETracker(args, frame_rate=30)

        # Keep last known image size for ByteTrack
        self.image_shape = (720, 1280)  # Default (height, width)

        self.get_logger().info(f"ByteTrack node subscribed to {self.input_topic}, publishing to {self.output_topic}")

    def detection_callback(self, msg: BoundingBoxes):
        detections = []
        class_map = {}

        # Estimate image size from boxes (optional, only if not fixed)
        for box in msg.boxes:
            x1, y1, x2, y2 = box.x_min, box.y_min, box.x_max, box.y_max
            conf = box.confidence
            class_name = box.class_name
            class_id = box.class_id
            depth = box.depth

            detections.append([x1, y1, x2, y2, conf, class_id, depth])
            class_map[class_id] = class_name

            # Update image shape dynamically from box coordinates
            self.image_shape = (
                max(self.image_shape[0], y2),
                max(self.image_shape[1], x2)
            )

        # ByteTrack expects image metadata
        img_info = self.image_shape  # (height, width)
        # img_size = (self.image_shape[1], self.image_shape[0])  # (width, height)
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
            class_name = class_map.get(class_id, "unknown") 

            tracked_box = TrackedBoundingBox()
            tracked_box.id = int(track_id)
            tracked_box.track_id = int(track_id)
            tracked_box.class_id = int(class_id)
            tracked_box.class_name = str(class_name)
            tracked_box.confidence = float(track.score)
            tracked_box.x_min = int(tlbr[0])
            tracked_box.y_min = int(tlbr[1])
            tracked_box.x_max = int(tlbr[2])
            tracked_box.y_max = int(tlbr[3])
            tracked_box.depth = float(track.depth)

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