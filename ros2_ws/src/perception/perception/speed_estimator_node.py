#!/usr/bin/env python3
import math
import cv2
import rclpy
from rclpy.node import Node
from msgs.msg import TrackedBoundingBoxes, TrackedBoundingBox

class SpeedEstimator:
    def __init__(self):
        # Store previous position and time for each track_id
        self.prev = {}  # track_id -> (x, y, z, t)

    def update(self, track_id: int, x: float, y: float, z: float, t: float):
        if track_id not in self.prev:
            self.prev[track_id] = (x, y, z, t)
            return 0.0, 0.0, 0.0, 0.0

        px, py, pz, pt = self.prev[track_id]
        dt = t - pt
        if dt <= 0.0:
            return 0.0, 0.0, 0.0, 0.0

        vx = (x - px) / dt
        vy = (y - py) / dt
        vz = (z - pz) / dt
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        self.prev[track_id] = (x, y, z, t)
        return vx, vy, vz, speed

class SpeedEstimationNode(Node):
    def __init__(self):
        super().__init__('speed_estimation_node')

        # Declare parameters
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('subscribe_topic', 'tracked_boxes_in')
        self.declare_parameter('publish_topic', 'tracked_boxes_with_speed')

        # Get parameters
        calib_path = self.get_parameter('calibration_file').get_parameter_value().string_value
        sub_topic = self.get_parameter('subscribe_topic').get_parameter_value().string_value
        pub_topic = self.get_parameter('publish_topic').get_parameter_value().string_value

        self.get_logger().info(f"Loading calibration file from {calib_path}")
        if not calib_path:
            self.get_logger().error('Parameter calib_file is empty; cannot load calibration.')
            rclpy.shutdown()
            return

        fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Cannot open calibration file: {calib_path}")

        # Read the projection matrix (3×4) for the left camera
        P = fs.getNode("projection_matrix_left").mat()  # numpy array shape (3,4)
        fs.release()

        # Grab fx, fy, cx, cy
        self.fx = float(P[0, 0])
        self.fy = float(P[1, 1])
        self.cx = float(P[0, 2])
        self.cy = float(P[1, 2])

        self.get_logger().info(f'Loaded intrinsics fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')

        # Initialize estimator
        self.estimator = SpeedEstimator()

        # Subscription and publisher
        self.sub = self.create_subscription(
            TrackedBoundingBoxes,
            sub_topic,
            self.cb_boxes,
            10
        )
        self.pub = self.create_publisher(
            TrackedBoundingBoxes,
            pub_topic,
            10
        )

    def cb_boxes(self, msg: TrackedBoundingBoxes):
        out = TrackedBoundingBoxes()
        out.header = msg.header
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        for box in msg.boxes:
            # Center pixel coord
            u = (box.x_min + box.x_max) * 0.5
            v = (box.y_min + box.y_max) * 0.5
            z = float(box.depth)

            # Project to camera-frame 3D
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy

            vx, vy, vz, speed = self.estimator.update(
                box.track_id, x, y, z, t
            )

            box.speed_x = float(vx)
            box.speed_y = float(vy)
            box.speed_z = float(vz)
            box.speed_mps = float(speed)
            out.boxes.append(box)

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SpeedEstimationNode()
    if rclpy.ok():
        rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()