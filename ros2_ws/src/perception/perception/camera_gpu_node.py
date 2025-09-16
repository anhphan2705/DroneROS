#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import ctypes
from pathlib import Path
import cv2

import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Optional VPI (GPU path)
try:
    import vpi
    HAVE_VPI = True
except Exception:
    HAVE_VPI = False


Gst.init(None)

# Load the shared lib
so_path = Path(__file__).parent / "libnvbuf_egl_bridge.so"
if not so_path.exists():
    raise FileNotFoundError(f"Couldn't find {so_path}. Put libnvbuf_egl_bridge.so in that folder.")
bridge = ctypes.CDLL(str(so_path))

# Declare the C function
bridge.gst_buffer_to_egl_image.argtypes = [ctypes.c_void_p]
bridge.gst_buffer_to_egl_image.restype = ctypes.c_void_p

def gst_buffer_to_vpi(buf):
    ptr = ctypes.c_void_p(hash(buf))   # raw GstBuffer* pointer for C side
    eglimg = bridge.gst_buffer_to_egl_image(ptr)
    if not eglimg:
        return None
    return vpi.asimage(eglimg, vpi.Format.NV12_ER, backend=vpi.Backend.CUDA)

class CameraGpuNode(Node):
    def __init__(self):
        super().__init__('camera_gpu_node')

        # ---------- Parameters ----------
        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 120)
        self.declare_parameter('udp_host', '192.168.0.254')
        self.declare_parameter('udp_port', 5600)
        self.declare_parameter('bitrate_kbps', 2000)

        self.W = int(self.get_parameter('width').value)
        self.H = int(self.get_parameter('height').value)
        self.FPS = int(self.get_parameter('fps').value)
        self.UDP_HOST = str(self.get_parameter('udp_host').value)
        self.UDP_PORT = int(self.get_parameter('udp_port').value)
        self.BITRATE = int(self.get_parameter('bitrate_kbps').value)

        self.hw = self.W // 2
        self.hh = self.H // 2

        # ---------- ROS pubs ----------
        self.bridge = CvBridge()
        self.pubs = [
            self.create_publisher(Image, '/camera0/image_raw', 10),
            self.create_publisher(Image, '/camera1/image_raw', 10),
            self.create_publisher(Image, '/camera2/image_raw', 10),
            self.create_publisher(Image, '/camera3/image_raw', 10),
        ]

        # ---------- Build GStreamer pipeline ----------
        # One capture → tee:
        # - branch 1: H.264 → RTP → UDP (for ground station)
        # - branch 2: NVMM appsink (for local GPU processing)
        pipeline_str = (
            "nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM),width={self.W},height={self.H},framerate={self.FPS}/1,format=NV12 ! "
            "tee name=t "
            # UDP branch
            "t. ! queue ! nvvidconv ! video/x-raw,format=I420 ! "
            f"x264enc bitrate={self.BITRATE} speed-preset=ultrafast tune=zerolatency key-int-max=5 threads=2 byte-stream=true ! "
            "rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={self.UDP_HOST} port={self.UDP_PORT} sync=false async=false "
            # Local NVMM branch
            "t. ! queue ! "
            "appsink name=ros_nvmm_sink emit-signals=true max-buffers=1 drop=true sync=false"
        )

        self.get_logger().info("GST pipeline:\n" + pipeline_str)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("ros_nvmm_sink")
        self.appsink.connect("new-sample", self._on_new_sample)

        self.pipeline.set_state(Gst.State.PLAYING)
        self.get_logger().info(f"Streaming UDP → {self.UDP_HOST}:{self.UDP_PORT} and publishing 4 topics.")

    # --------- NVMM → VPI helpers (fill in interop for your JetPack) ---------
    def _nvmm_to_vpiimage(self, sample) -> "vpi.Image|None":
        """
        Map an NVMM GstSample (NV12) into a VPIImage without copy.
        This tiny bridge depends on JetPack/VPI version. Two common patterns:
          - Use EGL interop: GstBuffer → EGLImage → vpi.asimage(eglImage, ...)
          - Use NvBufSurface → cudaEgl → vpi.asimage(...)
        Replace the TODO section with the 2-3 lines from the NVIDIA sample on your device.
        """
        if not HAVE_VPI:
            return None

        buf = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")

        # --- JP 6.0 EGL zero-copy path ---
        try:
            vpi_img = gst_buffer_to_vpi(buf)
            return vpi_img
        except Exception as e:
            self.get_logger().warn(f"EGL interop not available yet: {e}. Using CPU copy.")

        # Fallback to CPU mapping if interop isn't wired yet (keeps you running)
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            # NV12 in CPU for now (temporary): H*1.5 x W
            nv12 = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h * 3 // 2, w))
            # Wrap NV12 CPU into VPI to reuse same downstream code (one copy happened here)
            vpi_img = vpi.Image.from_numpy(nv12, format=vpi.Format.NV12_ER)
            return vpi_img
        finally:
            buf.unmap(mapinfo)

    def _emit_ros_image(self, cam_idx, bgr_np):
        msg = self.bridge.cv2_to_imgmsg(bgr_np, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"camera{cam_idx}"
        self.pubs[cam_idx].publish(msg)

    # --------- Appsink callback ---------
    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        # Prefer GPU path with VPI
        if HAVE_VPI:
            try:
                vpi_img = self._nvmm_to_vpiimage(sample)  # NV12 on GPU ideally
                if vpi_img is None:
                    raise RuntimeError("NVMM→VPI interop not set yet; using CPU fallback.")

                # Create 4 ROI views (no copy on GPU)
                rois = [
                    (0, 0, self.hw, self.hh),          # TL
                    (self.hw, 0, self.hw, self.hh),    # TR
                    (0, self.hh, self.hw, self.hh),    # BL
                    (self.hw, self.hh, self.hw, self.hh) # BR
                ]

                with vpi.Backend.CUDA:
                    for idx, (x, y, w, h) in enumerate(rois):
                        view = vpi_img.extract((x, y, w, h))
                        bgr_gpu = view.convert(vpi.Format.BGR8)
                        with bgr_gpu.cpu() as cpu_img:
                            bgr_np = cpu_img.as_numpy()
                            self._emit_ros_image(idx, bgr_np)
                return Gst.FlowReturn.OK
            except Exception as e:
                self.get_logger().warn(f"[VPI path] {e}; falling back to CPU split.")

        # -------- CPU fallback (still functional) --------
        try:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            w = caps.get_structure(0).get_value("width")
            h = caps.get_structure(0).get_value("height")
            ok, mapinfo = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.ERROR

            nv12 = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h * 3 // 2, w))
            # NV12 → BGR on CPU
            bgr = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

            half_h, half_w = h // 2, w // 2
            subs = [
                bgr[0:half_h,        0:half_w],        # TL
                bgr[0:half_h,        half_w:w],        # TR
                bgr[half_h:h,        0:half_w],        # BL
                bgr[half_h:h,        half_w:w],        # BR
            ]
            for i, sub in enumerate(subs):
                self._emit_ros_image(i, sub)

        finally:
            try:
                buf.unmap(mapinfo)
            except Exception:
                pass

        return Gst.FlowReturn.OK

    def destroy_node(self):
        try:
            self.pipeline.set_state(Gst.State.NULL)
        finally:
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


if __name__ == '__main__':
    main()