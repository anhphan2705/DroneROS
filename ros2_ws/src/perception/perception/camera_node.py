# camera_node.py
import rclpy
import cv2
import threading
from rclpy.node import Node
from cv_bridge import CvBridge
from queue import Queue
from sensor_msgs.msg import Image
from msgs.msg import RawCameraStatus

# GStreamer pipeline setup
def gstreamer_pipeline(
        capture_width=1920, 
        capture_height=1080, 
        display_width=1920, 
        display_height=1080, 
        framerate=30, 
        flip_method=0
    ):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class FrameReader(threading.Thread):
    def __init__(self, camera):
        threading.Thread.__init__(self)
        self.camera = camera
        self.queue = Queue(maxsize=1)
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                if not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                    except Exception:
                        pass
                self.queue.put(frame)

    def getFrame(self, timeout=None):
        return self.queue.get(timeout=timeout) if not self.queue.empty() else None

    def stop(self):
        self.running = False

class Camera:
    def __init__(self, node, width, height, framerate, flip_method):
        self.node = node
        self.flip_method = flip_method
        self.cap = cv2.VideoCapture(
            gstreamer_pipeline(
                capture_width=width, 
                capture_height=height, 
                display_width=width, 
                display_height=height, 
                framerate=framerate, 
                flip_method=flip_method
            ), 
            cv2.CAP_GSTREAMER
        )
        if not self.cap.isOpened():
            self.node.get_logger().error("Failed to open camera!")
            self.cap = None
            return
        self.frame_reader = FrameReader(self.cap)
        self.frame_reader.daemon = True
        self.frame_reader.start()

    def getFrame(self, timeout=None):
        if self.cap is None:
            return None
        return self.frame_reader.getFrame(timeout)

    def get_actual_settings(self):
        if self.cap is None:
            return None, None, None, None
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        return actual_width, actual_height, actual_fps, self.flip_method

    def close(self):
        if self.cap is not None:
            self.frame_reader.stop()
            self.cap.release()

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        self.declare_parameter('capture_width', 1920)
        self.declare_parameter('capture_height', 1080)
        self.declare_parameter('framerate', 30)
        self.declare_parameter('flip_method', 0)
        
        capture_width = self.get_parameter('capture_width').value
        capture_height = self.get_parameter('capture_height').value
        framerate = self.get_parameter('framerate').value
        flip_method = self.get_parameter('flip_method').value
        
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.status_publisher_ = self.create_publisher(RawCameraStatus, '/camera/status/raw_image_info', 10)  # Custom message publisher
        self.bridge = CvBridge()
        self.camera = Camera(self, capture_width, capture_height, framerate, flip_method)
        
        if self.camera.cap is None:
            self.get_logger().error("Camera initialization failed. Exiting node.")
            rclpy.shutdown()
            return
        
        self.timer = self.create_timer(0.1, self.publish_frame)  # 10 Hz
        self.status_timer = self.create_timer(1.0, self.publish_status)  # 1 Hz
        self.get_logger().info("Camera node started")

    def publish_frame(self):
        frame = self.camera.getFrame()
        if frame is not None:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)
        else:
            self.get_logger().warn("No frame received from camera.")

    def publish_status(self):
        actual_width, actual_height, actual_fps, flip_method = self.camera.get_actual_settings()
        if actual_width and actual_height and actual_fps:
            status_msg = RawCameraStatus()
            status_msg.framerate = actual_fps
            status_msg.image_width = actual_width
            status_msg.image_height = actual_height
            status_msg.flip_method = flip_method
            self.status_publisher_.publish(status_msg)
            self.get_logger().info(f"Published camera status: {status_msg}")

    def destroy_node(self):
        self.camera.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()