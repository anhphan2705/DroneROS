import rclpy
import time
import json
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class CameraSplitterNode(Node):
    def __init__(self):
        super().__init__('camera_splitter_node')

        self.subscriber = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        self.bridge = CvBridge()

        self.image_publishers = [
            self.create_publisher(Image, f'/camera/image_raw/split_{i}', 50)
            for i in range(4)
        ]
        self.status_publisher = self.create_publisher(String, '/camera/status/camera_splitter_status', 10)

        # FPS and size tracking
        self.frame_times = [time.time() for _ in range(4)]
        self.frame_sizes = [(0, 0) for _ in range(4)]
        self.last_fps_publish_time = time.time()  

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Get shape and define split points
            h, w, _ = cv_image.shape
            h_half, w_half = h // 2, w // 2

            quadrants = [
                cv_image[:h_half, :w_half],  # Top-left
                cv_image[:h_half, w_half:],  # Top-right
                cv_image[h_half:, :w_half],  # Bottom-left
                cv_image[h_half:, w_half:]   # Bottom-right
            ]

            for i in range(4):
                self.publish_image(i, quadrants[i], msg.header)

            # Publish FPS and size data every 30 seconds
            if time.time() - self.last_fps_publish_time >= 30:
                self.publish_fps_status()
                self.last_fps_publish_time = time.time()

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def publish_image(self, index, quadrant, header):
        image_msg = self.bridge.cv2_to_imgmsg(quadrant, encoding='bgr8')
        image_msg.header = header
        image_msg.header.frame_id = f"split_{index}"
        self.image_publishers[index].publish(image_msg)

        current_time = time.time()
        time_diff = current_time - self.frame_times[index]
        if time_diff > 0.01:
            self.frame_times[index] = current_time

        self.frame_sizes[index] = (quadrant.shape[1], quadrant.shape[0])  # width, height

    def publish_fps_status(self):
        current_time = time.time()
        fps_values = {
            f'split_{i}': round(1.0 / (current_time - t), 2) if (current_time - t) > 0.01 else 0
            for i, t in enumerate(self.frame_times)
        }
        size_values = {
            f'split_{i}': {'width': self.frame_sizes[i][0], 'height': self.frame_sizes[i][1]}
            for i in range(4)
        }
        status_data = {
            'fps': fps_values,
            'sizes': size_values
        }
        msg = String()
        msg.data = json.dumps(status_data)
        self.status_publisher.publish(msg)
        self.get_logger().info(f'Status: {status_data}')


def main(args=None):
    rclpy.init(args=args)
    node = CameraSplitterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
