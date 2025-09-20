#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2, gi, threading, time
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst

import vpi

class StitchStreamNode(Node):
    def __init__(self):
        super().__init__('stitch_stream_node')
        self.bridge = CvBridge()

        # ---- Parameters ----
        self.declare_parameter('udp_host', '192.168.0.254')
        self.declare_parameter('udp_port', 5700)
        self.declare_parameter('bitrate_kbps', 2500)
        self.declare_parameter('topics', [
            '/perception_img_visualizer_0',
            '/perception_img_visualizer_1',
            '/camera1/rectified',
            '/camera3/rectified'
        ])
        self.declare_parameter('out_width', 1280)
        self.declare_parameter('out_height', 720)
        self.declare_parameter('fps', 720)
        self.declare_parameter('output_mode', 'both')  # "udp", "ros", "both"
        
        self.UDP_HOST = self.get_parameter('udp_host').value
        self.UDP_PORT = self.get_parameter('udp_port').value
        self.BITRATE  = self.get_parameter('bitrate_kbps').value
        self.topics   = self.get_parameter('topics').value
        self.OUT_W    = self.get_parameter('out_width').value
        self.OUT_H    = self.get_parameter('out_height').value
        self.FPS      = self.get_parameter('fps').value
        self.MODE     = self.get_parameter('output_mode').value.lower()
        
        self.frames = {t: None for t in self.topics}
        
        # ROS publisher for stitched image
        if self.MODE in ("ros", "both"):
            self.stitched_pub = self.create_publisher(Image, "/stitched/image", 10)
        else:
            self.stitched_pub = None

        # Subscribers
        for t in self.topics:
            self.create_subscription(Image, t, self.make_cb(t), 10)

        # Setup GStreamer pipeline (x264enc → RTP → UDP)
        pipeline_str = (
            f"appsrc name=appsrc is-live=true block=true format=3 "
            f"caps=video/x-raw,format=BGR,width={self.OUT_W},height={self.OUT_H},framerate={self.FPS}/1 ! "
            "videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc bitrate={self.BITRATE} speed-preset=ultrafast tune=zerolatency key-int-max=5 ! "
            "rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={self.UDP_HOST} port={self.UDP_PORT} sync=false async=false"
        )
        
        self.get_logger().info(f"Pipeline:\n{pipeline_str}")
        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsrc = self.pipeline.get_by_name("appsrc")
        self.pipeline.set_state(Gst.State.PLAYING)

        # Worker thread
        self._stop = False
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()

    def make_cb(self, topic):
        def cb(msg):
            try:
                frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                self.frames[topic] = frame
            except Exception as e:
                self.get_logger().error(f"Failed to convert image from {topic}: {e}")
        return cb

    def vpi_resize(self, img, w, h):
        # CPU numpy → VPI
        with vpi.Backend.PVA:
            vpi_in = vpi.asimage(img)
            vpi_resized = vpi_in.resize((w, h))
            return vpi_resized.cpu()

    def loop(self):
        while not self._stop and rclpy.ok():
            # Require all 4 frames
            if any(self.frames[t] is None for t in self.topics):
                time.sleep(0.01)
                continue

            try:
                # Compute quadrant size
                q_w = self.OUT_W // 2
                q_h = self.OUT_H // 2

                # Resize quadrants
                resized = []
                for t in self.topics:
                    r = self.vpi_resize(self.frames[t], q_w, q_h)
                    resized.append(r)

                # Stitch into one frame
                top = np.hstack((resized[0], resized[1]))
                bot = np.hstack((resized[2], resized[3]))
                stitched = np.vstack((top, bot))

                # ---- Publish to ROS ----
                if self.stitched_pub:
                    img_msg = self.bridge.cv2_to_imgmsg(stitched, encoding='bgr8')
                    img_msg.header.stamp = self.get_clock().now().to_msg()
                    img_msg.header.frame_id = "stitched"
                    self.stitched_pub.publish(img_msg)

                # ---- Push to GStreamer ----
                if self.appsrc:
                    buf = Gst.Buffer.new_wrapped(stitched.tobytes())
                    self.appsrc.emit("push-buffer", buf)

            except Exception as e:
                self.get_logger().warn(f"Stitch error: {e}")
                continue

    def destroy_node(self):
        self._stop = True
        if self.worker.is_alive():
            self.worker.join(timeout=1.0)
        self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def main():
    rclpy.init()
    node = StitchStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
