import rclpy
import threading
import time
from rclpy.node import Node
from sensor_msgs.msg import Image
from msgs.msg import AutofocusStatus
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import cv2
import numpy as np
from perception.focus_tools.focuser import Focuser
from perception.focus_tools.autofocus import doFocus, FocusState, laplacian, FrameStorage

class AutofocusNode(Node):
    def __init__(self):
        super().__init__('autofocus_node')
        
        # Declare ROS 2 parameters
        self.declare_parameter('i2c_bus', 9)
        self.declare_parameter('focus_interval', 5.0)
        
        # Read parameters
        i2c_bus = self.get_parameter('i2c_bus').value
        self.focus_interval = self.get_parameter('focus_interval').value

        self.image_publisher_ = self.create_publisher(Image, '/camera/autofocus_image', 10)
        self.info_publisher_ = self.create_publisher(AutofocusStatus, '/camera/status/autofocus_info', 10)
        self.bridge = CvBridge()
        self.focuser = Focuser(i2c_bus)
        self.focus_state = FocusState()
        self.frame_storage = FrameStorage()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("Autofocus node started")

        # Initialize FPS measurement
        self.last_frame_time = None
        self.last_autofocus_time = None
        self.output_fps = 0.0

        # Handle ROS 2 shutdown properly
        rclpy.get_default_context().on_shutdown(self.cleanup)

    def image_callback(self, msg):
        """Update FrameStorage with the latest frame from /camera/image_raw."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frame_storage.update_frame(frame)

        # Decide if we should doFocus
        now = time.time()
        if self.last_autofocus_time is None or (now - self.last_autofocus_time) > self.focus_interval:
            # Potentially check a sharpness threshold here,
            # or just run doFocus every focus_interval
            self.run_autofocus()
            self.last_autofocus_time = now

    def run_autofocus(self):
        """Calls doFocus() with our frame_storage as 'camera'."""
        if self.focus_state.isFinish():
            self.focus_state.reset()
            doFocus(self.frame_storage, self.focuser, self.focus_state)
            max_focus = self.focuser.read()

            # Optionally publish status
            status_msg = AutofocusStatus()
            status_msg.focus_setting = max_focus
            self.info_publisher_.publish(status_msg)

            self.get_logger().info(f"Autofocus done. Setting focus to {max_focus}")
        else:
            self.get_logger().warning("Autofocus triggered. Previous command not done!")

    # def autofocus_callback(self, msg):
    #     """Processes incoming images and triggers autofocus only when needed."""
    #     current_time = time.time()
    #     frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    #     ros_image = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
    #     self.image_publisher_.publish(ros_image)
    #     # sharpness = laplacian(frame)
    
    #     # Measure output FPS
    #     if self.last_frame_time is not None:
    #         self.output_fps = 1.0 / (current_time - self.last_frame_time)
    #     self.last_frame_time = current_time
    
    #     # Check if autofocus is needed based on time interval
    #     if self.last_autofocus_time is None or (current_time - self.last_autofocus_time) > self.focus_interval:
    #         # if sharpness < self.focus_state.threshold: # TODO: Find a good threshold to futher optimize
    #         if self.focus_state.isFinish():
    #             self.focus_state.reset()
    #             doFocus(None, self.focuser, self.focus_state)
    #             max_focus = self.focuser.read()
    #             self.get_logger().info(f"Autofocus triggered. Setting focus to {max_focus}")
    
    #             # Update last autofocus time
    #             self.last_autofocus_time = current_time
    
    #             # Publish Autofocus Status
    #             info_msg = AutofocusStatus()
    #             info_msg.focus_setting = max_focus
    #             info_msg.image_width = frame.shape[1]
    #             info_msg.image_height = frame.shape[0]
    #             info_msg.output_framerate = self.output_fps
    #             self.info_publisher_.publish(info_msg)
    #         else:
    #             self.get_logger().warning(f"Autofocus triggered. Previous Autofocus command not done.")

    def cleanup(self):
        self.get_logger().info("Shutting down AutofocusNode...")


def main():
    rclpy.init()
    node = AutofocusNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
