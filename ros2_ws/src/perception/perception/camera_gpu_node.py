#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# VPI (GPU ops)
import vpi

class CameraGpuNode(Node):
    def __init__(self):
        super().__init__('camera_gpu_node')

        # ---- Parameters ----
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 60)
        self.declare_parameter('udp_host', '192.168.0.254')
        self.declare_parameter('udp_port', 5600)
        self.declare_parameter('bitrate_kbps', 2000)

        self.W = self.get_parameter('width').value
        self.H = self.get_parameter('height').value
        self.FPS = self.get_parameter('fps').value
        self.UDP_HOST = self.get_parameter('udp_host').value
        self.UDP_PORT = self.get_parameter('udp_port').value
        self.BITRATE = self.get_parameter('bitrate_kbps').value

        self.bridge = CvBridge()
        self.pubs = [
            self.create_publisher(Image, f'/camera{i}/image_raw', 10)
            for i in range(4)
        ]

        # ---- GStreamer pipeline ----
        pipeline_str = (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM),width={self.W},height={self.H},"
            f"framerate={self.FPS}/1,format=NV12 ! "
            "tee name=t "
            # Branch 1: UDP
            "t. ! queue ! nvvidconv ! video/x-raw,format=I420 ! "
            f"x264enc bitrate={self.BITRATE} speed-preset=ultrafast tune=zerolatency key-int-max=5 ! "
            "rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={self.UDP_HOST} port={self.UDP_PORT} sync=false async=false "
            # # Branch 2: appsink for GPU processing
            # "t. ! queue ! appsink name=ros_sink emit-signals=true max-buffers=1 drop=true sync=false"
            # Local CPU branch
            "t. ! queue ! nvvidconv ! video/x-raw,format=NV12 ! "
            "appsink name=ros_sink emit-signals=true max-buffers=1 drop=true sync=false"
        )

        self.get_logger().info(f"Pipeline:\n{pipeline_str}")
        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("ros_sink")
        self.vpi_stream = vpi.Stream()
        self.appsink.connect("new-sample", self.on_new_sample)
        self.pipeline.set_state(Gst.State.PLAYING)
        
    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok or len(mapinfo.data) < w * h * 3 // 2:
            self.get_logger().warn(f"Got small buffer ({len(mapinfo.data)} bytes), skipping frame")
            return Gst.FlowReturn.OK

        try:
            # NV12 in CPU memory: Y (h*w) + interleaved UV (h/2*w)
            nv12 = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h * 3 // 2, w))
            y_plane  = nv12[0:h, :]
            uv_plane = nv12[h:, :].reshape(h // 2, w // 2, 2)

            # Wrap CPU planes into a VPI image with explicit NV12 (even-range) format
            vpi_host = vpi.asimage([y_plane, uv_plane], vpi.Format.NV12_ER)

            # All heavy work on GPU
            with vpi.Backend.CUDA:
                # NV12 -> BGR8 on CUDA (queued on our stream)
                vpi_bgr = vpi_host.convert(vpi.Format.BGR8, stream=self.vpi_stream)

                # Prepare 4 even-sized quadrants (NV12/BGR8 prefer even geometry)
                half_w = (w // 2) & ~1
                half_h = (h // 2) & ~1
                rects = (
                    vpi.RectangleI(0,       0,       half_w, half_h),  # TL
                    vpi.RectangleI(half_w,  0,       half_w, half_h),  # TR
                    vpi.RectangleI(0,       half_h,  half_w, half_h),  # BL
                    vpi.RectangleI(half_w,  half_h,  half_w, half_h),  # BR
                )
                rois = [vpi.Image.view(vpi_bgr, r) for r in rects]

            # Sync once so all CUDA ops finish before CPU downloads
            self.vpi_stream.sync()

            # Now, per-ROI: one download + publish
            now = self.get_clock().now().to_msg()
            for i, roi in enumerate(rois):
                bgr_np = roi.cpu()   # directly np.ndarray
                msg = self.bridge.cv2_to_imgmsg(bgr_np, encoding='bgr8')
                msg.header.stamp = now
                msg.header.frame_id = f"camera{i}"
                self.pubs[i].publish(msg)

        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def destroy_node(self):
        self.pipeline.set_state(Gst.State.NULL)
        super().destroy_node()


def main():
    rclpy.init()
    node = CameraGpuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
