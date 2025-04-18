#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
from msgs.msg import BoundingBox, BoundingBoxes
import torch
import os


class YOLOv8InferenceNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Declare and get parameters
        self.declare_parameter('model_path', 'best.pt')
        self.declare_parameter('image_topic', '/camera/rectified/split_0')
        self.declare_parameter('detection_topic', '/yolo/detections')

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        if not os.path.isabs(self.model_path):
            package_path = get_package_share_directory('perception')
            self.model_path = os.path.join(package_path, 'models', self.model_path)
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found at: {self.model_path}")
            return
        self.get_logger().info(f"Using model: {self.model_path}")

        # Determine extension and load model
        ext = os.path.splitext(self.model_path)[1].lower()
        if ext == '.pt':
            # PyTorch model: can train, val, predict, export
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(self.model_path).to(self.device)
        elif ext == '.onnx':
            # ONNX model: only supports predict/val; specify task and let ORT handle providers
            self.device = 0 if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(self.model_path, task='detect')
        else:
            self.get_logger().error(f"Unsupported model format: {ext}")
            return

        # Load class names
        self.class_names = self.model.names
        self.get_logger().info(f"Loaded model on device={self.device}, classes={self.class_names}")

        # ROS topics
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value

        # ROS interfaces
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)
        self.detection_pub = self.create_publisher(
            BoundingBoxes, self.detection_topic, 10)

        self.get_logger().info("YOLOv8 Inference Node initialized.")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CVBridge error: {e}")
            return

        boxes_msg = BoundingBoxes()
        boxes_msg.header = msg.header

        # Unified predict for both model types
        results = self.model.predict(
            frame,
            imgsz=640,
            device=self.device,
            verbose=False
        )[0]

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Skip invalid class indices
            if cls < 0 or cls >= len(self.class_names):
                self.get_logger().warn(f"Skipping detection with invalid class id {cls}")
                continue

            det = BoundingBox()
            det.id = i
            det.class_id = cls
            det.class_name = self.class_names[cls]
            det.confidence = conf
            det.x_min = x1
            det.y_min = y1
            det.x_max = x2
            det.y_max = y2
            det.depth = -1.0
            boxes_msg.boxes.append(det)

        self.get_logger().info(
            f"[{len(boxes_msg.boxes)}] detections at t={msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
        )
        self.detection_pub.publish(boxes_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()