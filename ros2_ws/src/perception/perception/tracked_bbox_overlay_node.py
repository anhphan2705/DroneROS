#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2, socket, json
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
        self.declare_parameter('udp_host', '192.168.0.254')
        self.declare_parameter('udp_port', 6000)
        self.declare_parameter('source_id', 0)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        tracked_topic = self.get_parameter('tracked_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        udp_host     = self.get_parameter('udp_host').value
        udp_port     = self.get_parameter('udp_port').value
        self.source_id    = self.get_parameter('source_id').value
        
        # ---- Networking ----
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_addr = (udp_host, udp_port)

        # ---- ROS I/O ----
        self.bridge = CvBridge()

        self.image_sub = Subscriber(self, Image, image_topic)
        self.tracked_sub = Subscriber(self, TrackedBoundingBoxes, tracked_topic)

        self.ts = ApproximateTimeSynchronizer([self.image_sub, self.tracked_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.callback)

        self.image_pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f"TrackedBBoxOverlayNode started — drawing on image + broadcasting over {udp_host}.")

    def callback(self, img_msg: Image, tracked_msg: TrackedBoundingBoxes):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # JSON metadata payload
        boxes_json = {
            "stamp": img_msg.header.stamp.sec,
            "frame_id": img_msg.header.frame_id,
            "source_id": self.source_id,
            "boxes": []
        }

        for box in tracked_msg.boxes:
            x1, y1, x2, y2 = box.x_min, box.y_min, box.x_max, box.y_max
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)

            label = f"tID: {box.track_id} | ID: {box.class_name}:{box.id:02d} \n"
            # label += f" | Conf: {box.confidence_interval[i]:.2f}"
            if box.classification_id > 0:
                label += f' | cID: {box.classification_id:.2f} |'
            if box.depth > 0:
                label += f' ({box.depth:.1f}m)'
            if box.speed_mps > 0:
                label += f' ({box.speed_mps:.1f}m/s)'
            
            lines = label.split('\n')
            font     = cv2.FONT_HERSHEY_SIMPLEX
            scale    = 0.5
            thickness= 1
            line_h   = int(30 * scale)
            for i, line in enumerate(reversed(lines)):
                y = y1 - 8 - (i * line_h)
                cv2.putText(frame, line, (x1, y), font, scale, (0,0,0), thickness, cv2.LINE_AA)
                
            # Add to JSON packet
            boxes_json["boxes"].append({
                "track_id": box.track_id,
                "class_id": box.id,
                "class_name": box.class_name,
                "x_min": box.x_min,
                "y_min": box.y_min,
                "x_max": box.x_max,
                "y_max": box.y_max,
                "confidence": getattr(box, "confidence_interval", [0])[0] if hasattr(box, "confidence_interval") else None,
                "classification_id": getattr(box, "classification_id", -1),
                "depth": getattr(box, "depth", -1.0),
                "speed_mps": getattr(box, "speed_mps", -1.0)
            })

        self.get_logger().debug(f"Overlayed {len(tracked_msg.boxes)} tracked boxes on image.")
        overlay_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        overlay_msg.header = img_msg.header
        self.image_pub.publish(overlay_msg)
        
        # ---- Send JSON metadata (UDP) ----
        try:
            payload = json.dumps(boxes_json).encode('utf-8')
            if len(payload) > 1400:
                self.get_logger().warn(f"UDP payload too large ({len(payload)} bytes), truncating")
                payload = payload[:1400]
            self.sock.sendto(payload, self.udp_addr)
        except Exception as e:
            self.get_logger().error(f"UDP send error: {e}")
            
    def destroy_socket(self):
        self.sock.close()


def main(args=None):
    rclpy.init(args=args)
    node = TrackedBBoxOverlayNode()
    rclpy.spin(node)
    node.destroy_socket()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()