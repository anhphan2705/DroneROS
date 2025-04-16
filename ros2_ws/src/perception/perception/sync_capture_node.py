#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import datetime
import threading
from message_filters import ApproximateTimeSynchronizer, Subscriber
from msgs.srv import CaptureImageRequest

class SyncCaptureNode(Node):
    def __init__(self):
        super().__init__('sync_capture_node')
        # Set logger to debug level (you can also use command-line args: --ros-args --log-level DEBUG)
        # self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
        self.bridge = CvBridge()

        # Main folder to save screenshots
        self.save_directory = 'image_captured'
        os.makedirs(self.save_directory, exist_ok=True)

        # Define camera IDs and topics
        self.camera_ids = ['camera0', 'camera1', 'camera2', 'camera3']
        self.topics = {
            'camera0': '/camera/image_raw/split_0',
            'camera1': '/camera/image_raw/split_1',
            'camera2': '/camera/image_raw/split_2',
            'camera3': '/camera/image_raw/split_3'
        }

        # Create a subfolder for each camera.
        self.camera_folders = {}
        for cam in self.camera_ids:
            cam_folder = os.path.join(self.save_directory, cam)
            os.makedirs(cam_folder, exist_ok=True)
            self.camera_folders[cam] = cam_folder
            self.get_logger().debug(f"Created folder {cam_folder} for {cam}")

        # Create message_filters subscribers for each camera topic.
        self.subscribers = {}
        for cam in self.camera_ids:
            self.subscribers[cam] = Subscriber(self, Image, self.topics[cam])
            self.get_logger().debug(f"Subscribed to {self.topics[cam]} for {cam}")

        # Set up an approximate time synchronizer for all camera subscribers.
        self.sync = ApproximateTimeSynchronizer(list(self.subscribers.values()),
                                                queue_size=10,
                                                slop=0.05)
        self.sync.registerCallback(self.sync_callback)

        # This variable holds the last synchronized set of images (as ROS messages).
        self.last_sync = {}
        # Event to signal that a new synchronized set is available.
        self.sync_event = threading.Event()

        # Create a service to trigger the synchronized screenshot capture.
        self.srv = self.create_service(CaptureImageRequest,
                                       'capture_sync_image',
                                       self.service_callback)

        self.get_logger().info("SyncCaptureNode initialized and ready.")

    def sync_callback(self, *msgs):
        """
        Callback from the synchronizer.
        msgs is a tuple of Image messages in the order of self.camera_ids.
        """
        self.get_logger().debug("Received synchronized messages.")
        sync_dict = {}
        for i, cam in enumerate(self.camera_ids):
            sync_dict[cam] = msgs[i]
            self.get_logger().debug(f"Camera: {cam}, Timestamp: {msgs[i].header.stamp}")
        self.last_sync = sync_dict
        # Signal that a new synchronized set is available.
        self.sync_event.set()

    def service_callback(self, request, response):
        self.get_logger().info("Service callback triggered. Checking for recent synchronized set.")
        
        # Here you might store the time of the last synchronized set (e.g., in self.last_sync_time)
        # For simplicity, let’s assume you decide to wait with a longer timeout
        if not self.sync_event.wait(timeout=1.0):
            response.success = False
            response.message = "Timeout waiting for synchronized images."
            self.get_logger().debug("Timeout waiting for synchronized images.")
            return response

        # Optionally, you can check if the synchronized set is recent enough.
        # For example:
        # if (current_time - self.last_sync_time) > threshold:
        #     response.success = False
        #     response.message = "Synchronized set too old."
        #     return response

        # Now process the latest synchronized set
        self.sync_event.clear()

        captured = []
        missing = []
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.get_logger().info(f"Synchronized set received. Processing capture with timestamp: {timestamp}")

        for cam_id in request.camera_ids:
            if cam_id in self.last_sync:
                msg = self.last_sync[cam_id]
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    filename = os.path.join(self.camera_folders[cam_id],
                                            f"{cam_id}_sync_{timestamp}.png")
                    cv2.imwrite(filename, cv_image)
                    captured.append(cam_id)
                    self.get_logger().debug(f"Saved {cam_id} image to {filename}")
                except CvBridgeError as e:
                    missing.append(cam_id)
                    self.get_logger().error(f"Conversion error for {cam_id}: {e}")
            else:
                missing.append(cam_id)
                self.get_logger().warn(f"Camera ID {cam_id} not in synchronized set.")

        if captured:
            response.success = True
            response.message = f"Captured images for: {', '.join(captured)}."
            if missing:
                response.message += f" Failed for: {', '.join(missing)}."
        else:
            response.success = False
            response.message = "No images captured."
        
        self.get_logger().info(f"Service response: {response}")
        return response

def main(args=None):
    rclpy.init(args=args)
    node = SyncCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
    
    
# How to use it: ros2 service call /capture_sync_image msgs/srv/CaptureImageRequest "{camera_ids: ['camera0', 'camera2']}"