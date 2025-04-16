#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

import cv2
import numpy as np
import torch
import os
from ultralytics import YOLO
import onnxruntime as ort

from msgs.msg import BoundingBox, BoundingBoxes


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
        else:
            self.get_logger().info(f"Using model with path: {self.model_path}")   

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.model_type = os.path.splitext(self.model_path)[-1]
        self.class_names = []

        # Load model based on extension
        if self.model_type == '.pt':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(self.model_path).to(self.device)
            if not hasattr(self.model, 'predict'):
                self.get_logger().error("YOLOv8 PyTorch model failed to load")
            self.class_names = self.model.names
            self.get_logger().info(f"Loaded PyTorch model on {self.device}")
            self.get_logger().info(f"Loaded class names: {self.class_names} ...")
            
        elif self.model_type == '.onnx':
            providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            if not self.session.get_inputs():
                self.get_logger().error("ONNX model failed to load — no inputs found.")
            self.get_logger().info(f"Loaded ONNX model with providers: {providers}")
            self.get_logger().info(f"ONNX input: {self.input_name}, shape: {self.session.get_inputs()[0].shape}")
        else:
            raise ValueError("Unsupported model format. Use .pt or .onnx")

        # Set up ROS interfaces
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.detection_pub = self.create_publisher(BoundingBoxes, self.detection_topic, 10)

        self.get_logger().info("YOLOv8 Inference Node initialized.")

    def preprocess_onnx(self, img):
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_norm, (2, 0, 1))
        img_input = np.expand_dims(img_input, axis=0)
        return img_input

    def postprocess_onnx(self, outputs, original_shape):
        boxes, scores, class_ids = [], [], []
        h, w = original_shape
        detections = outputs[0][0]

        for det in detections:
            conf = det[4]
            if conf > 0.25:
                cls_scores = det[5:]
                cls_id = np.argmax(cls_scores)
                score = cls_scores[cls_id] * conf
                if score > 0.25:
                    cx, cy, bw, bh = det[:4]
                    x1 = int((cx - bw / 2) * w / 640)
                    y1 = int((cy - bh / 2) * h / 640)
                    x2 = int((cx + bw / 2) * w / 640)
                    y2 = int((cy + bh / 2) * h / 640)
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    class_ids.append(cls_id)
        return boxes, scores, class_ids

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CVBridge error: {e}")
            return

        boxes_msg = BoundingBoxes()
        boxes_msg.header = msg.header

        if self.model_type == '.pt':
            results = self.model.predict(frame, imgsz=640, device=self.device, verbose=False)[0]
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                det = BoundingBox()
                det.id = i
                det.class_name = self.class_names[cls]
                det.confidence = conf
                det.x_min = x1
                det.y_min = y1
                det.x_max = x2
                det.y_max = y2
                boxes_msg.boxes.append(det)

        elif self.model_type == '.onnx':
            img_input = self.preprocess_onnx(frame)
            outputs = self.session.run(None, {self.input_name: img_input})
            boxes, scores, class_ids = self.postprocess_onnx(outputs, frame.shape[:2])

            for i, ((x1, y1, x2, y2), score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
                det = BoundingBox()
                det.id = i
                det.class_name = self.class_names[cls_id]
                det.confidence = score
                det.x_min = x1
                det.y_min = y1
                det.x_max = x2
                det.y_max = y2
                boxes_msg.boxes.append(det)
        
        self.get_logger().info(f"[{len(boxes_msg.boxes)}] detections at t={msg.header.stamp.sec}.{msg.header.stamp.nanosec}")
        self.detection_pub.publish(boxes_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()