#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs.msg import TrackedBoundingBoxes, TrackedBoundingBox
from message_filters import Subscriber, ApproximateTimeSynchronizer
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
import torch
from torchvision import transforms
from PIL import Image as PILImage
import cv2
from ultralytics import YOLO

class ClassificationNode(Node):
    def __init__(self):
        super().__init__('classification_node')

        # Parameters
        self.declare_parameter('tracked_topic', '/yolo/detections_1/depth/tracked')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('classifier_model_path', 'sign_classification_model.pt')
        self.declare_parameter('input_size', [0, 0]) # [H, W] or [0, 0] for dynamic
        self.declare_parameter('class_ids_to_classify', [-1]) # list of int class_ids or -1 to classify all
        self.declare_parameter('classification_topic', '/yolo/detections_1/depth/tracked/classified')

        self.tracked_topic = self.get_parameter('tracked_topic').value
        self.image_topic = self.get_parameter('image_topic').value
        model_path_param = self.get_parameter('classifier_model_path').value
        self.class_ids_to_classify = self.get_parameter('class_ids_to_classify').value or []
        self.input_size = self.get_parameter('input_size').value or [0, 0]
        self.classification_topic = self.get_parameter('classification_topic').value

        if self.class_ids_to_classify == [-1]:
            self.class_ids_to_classify = []
            
        # Resolve model path
        if os.path.isabs(model_path_param):
            self.model_path = model_path_param
        else:
            pkg_dir = get_package_share_directory('perception')
            self.model_path = os.path.join(pkg_dir, "models", model_path_param)
        if not os.path.exists(self.model_path):
            self.get_logger().error(f"Model file not found: {self.model_path}")
            rclpy.shutdown()
            return

        # CV bridge for image conversion
        self.bridge = CvBridge()

        # Load classifier model
        try:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            self.model = YOLO(self.model_path).to(self.device)
            self.class_names = self.model.names
            self.get_logger().info(f"Loaded model on device={self.device}, classes={self.class_names}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLOv8 model: {e}")
            return

        # Build preprocessing transforms
        transforms_list = []
        # Only resize if user supplied positive dimensions
        if isinstance(self.input_size, list) and len(self.input_size) == 2 \
           and self.input_size[0] > 0 and self.input_size[1] > 0:
            transforms_list.append(transforms.Resize(tuple(self.input_size)))
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.preprocess = transforms.Compose(transforms_list)
        dynamic_support = not (self.input_size[0] > 0 and self.input_size[1] > 0)

        # Subscribers & synchronization
        img_sub = Subscriber(self, Image, self.image_topic)
        trk_sub = Subscriber(self, TrackedBoundingBoxes, self.tracked_topic)
        ats = ApproximateTimeSynchronizer([img_sub, trk_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)

        # Publisher for classified tracked boxes
        self.pub = self.create_publisher(TrackedBoundingBoxes, self.classification_topic, 10)

        self.get_logger().info(
            f"Classification node ready. Using {self.device}.\n"
            f" Subscribed to: {self.tracked_topic}\n"
            f" and {self.image_topic}\n"
            f"Class IDs to classify: {self.class_ids_to_classify if self.class_ids_to_classify else 'ALL'}\n"
            f"Dynamic resize: {'enabled' if dynamic_support else 'disabled'}"
        )

    def callback(self, img_msg: Image, trk_msg: TrackedBoundingBoxes):
        # Convert ROS image to OpenCV
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return
        h, w = cv_img.shape[:2]

        crops, ids = [], []
        # Collect ROIs only for specified class_ids (or all if empty)
        for box in trk_msg.boxes:
            if self.class_ids_to_classify and box.class_id not in self.class_ids_to_classify:
                continue
            x1, y1 = max(0, box.x_min), max(0, box.y_min)
            x2, y2 = min(w, box.x_max), min(h, box.y_max)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = cv_img[y1:y2, x1:x2]
            pil_img = PILImage.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            crops.append(tensor)
            ids.append(box.track_id)

        # If no crops, publish defaults (-1)
        if not crops:
            out_msg = TrackedBoundingBoxes()
            out_msg.header = trk_msg.header
            for box in trk_msg.boxes:
                new_box = TrackedBoundingBox()
                new_box.id = box.id
                new_box.track_id = box.track_id
                new_box.class_id = box.class_id
                new_box.classification_id = -1
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
                out_msg.boxes.append(new_box)
            self.pub.publish(out_msg)
            self.get_logger().debug(f"Publishing {len(ids)} classification IDs.")
            return

        # Batch inference for collected ROIs
        batch = torch.cat(crops, dim=0)
        with torch.no_grad():
            try:
                output = self.model.model(batch)         # Tensor [B, C]
                raw_logits = output[0] if isinstance(output, (tuple, list)) else output
            except Exception as e:
                self.get_logger().error(f"Inference failed: {e}")
                return
            preds = torch.argmax(raw_logits, dim=1).cpu().numpy().tolist()
        id_to_cls = {tid: cls for tid, cls in zip(ids, preds)}

        # Build and publish output
        out_msg = TrackedBoundingBoxes()
        out_msg.header = trk_msg.header
        for box in trk_msg.boxes:
            new_box = TrackedBoundingBox()
            new_box.id = box.id
            new_box.track_id = box.track_id
            new_box.class_id = box.class_id
            new_box.classification_id = id_to_cls.get(box.track_id, -1)
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
            out_msg.boxes.append(new_box)
        self.pub.publish(out_msg)
        self.get_logger().info(f"Publishing {len(id_to_cls)} classification IDs.")


def main(args=None):
    rclpy.init(args=args)
    node = ClassificationNode()
    if rclpy.ok():
        rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()