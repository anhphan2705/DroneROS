#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import gi
import cv2

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# VPI (GPU ops)
import vpi

def load_calibration(calib_file: str):
    """
    Load rectification maps + intrinsics from a stereo calibration file.

    Returns:
        map_left   (np.ndarray HxWx2, float32)
        map_right  (np.ndarray HxWx2, float32)
        fx         (float) focal length in px (rectified)
        baseline_m (float) baseline in meters (rectified)
    """
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open calibration file {calib_file}")

    # --- Rectification maps ---
    def read_side(prefix_xy):
        m1 = fs.getNode(f"{prefix_xy}_x").mat()
        m2 = fs.getNode(f"{prefix_xy}_y").mat()
        if m1 is None or m2 is None:
            raise RuntimeError(f"Missing {prefix_xy}_x or {prefix_xy}_y in {calib_file}")

        if m1.ndim == 3 and m1.dtype == np.int16:
            map2f, _ = cv2.convertMaps(m1, m2, cv2.CV_32FC2)
            return np.ascontiguousarray(map2f, dtype=np.float32)

        if m1.ndim == 2 and m1.dtype == np.float32 and m2.ndim == 2 and m2.dtype == np.float32:
            map2f = np.dstack([m1, m2]).astype(np.float32)
            return np.ascontiguousarray(map2f, dtype=np.float32)

        if m1.ndim == 3 and m1.dtype == np.float32 and m1.shape[2] == 2:
            return np.ascontiguousarray(m1, dtype=np.float32)

        raise RuntimeError(f"Unsupported map formats for {prefix_xy}")

    map_left  = read_side('stereo_map_left')
    map_right = read_side('stereo_map_right')

    # --- Projection matrices (rectified) ---
    P1 = fs.getNode('projection_matrix_left').mat()
    P2 = fs.getNode('projection_matrix_right').mat()
    fs.release()

    if P1 is None or P2 is None:
        raise RuntimeError("Calibration file missing projection_matrix_left or projection_matrix_right")

    fx = float(P1[0, 0])  # rectified focal length
    Tx = float(P2[0, 3])  # usually = -fx * baseline
    baseline_m = abs(Tx) / fx

    return map_left, map_right, fx, baseline_m

# This version manual cropping out odd pixel column to fit the 540x960
def build_vpi_warp(map_x: np.ndarray, map_y: np.ndarray):
    """
    Build a WarpMap using calibration maps. Handles VPI padding.
    """
    assert map_x.shape == map_y.shape
    h, w = map_x.shape

    grid = vpi.WarpGrid((w, h))  # VPI may pad internally
    warp = vpi.WarpMap(grid)
    arr = np.asarray(warp)

    # Copy only overlapping region
    arr[:h, :w, 0] = map_x
    arr[:h, :w, 1] = map_y

    return warp


class CameraGpuNode(Node):
    def __init__(self):
        super().__init__('camera_gpu_node')

        # ---- Parameters ----
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('fps', 60)
        self.declare_parameter('udp_host', '192.168.0.254')
        self.declare_parameter('udp_port', 5600)
        self.declare_parameter('bitrate_kbps', 2000)
        self.declare_parameter('calibration_file_0', '')
        self.declare_parameter('calibration_file_1', '')

        self.W = self.get_parameter('width').value
        self.H = self.get_parameter('height').value
        self.FPS = self.get_parameter('fps').value
        self.UDP_HOST = self.get_parameter('udp_host').value
        self.UDP_PORT = self.get_parameter('udp_port').value
        self.BITRATE = self.get_parameter('bitrate_kbps').value
        
        # Load 2-channel float32 absolute-XY maps at ROI size (half-res)
        L0_2f, R0_2f, self.fx, self.baseline_m= load_calibration(self.get_parameter('calibration_file_0').value)
        L1_2f, R1_2f, _, _ = load_calibration(self.get_parameter('calibration_file_1').value)

        # Keep NumPy alive and wrap as VPI images (format: 2-channel float32)
        self.warp_left_0  = build_vpi_warp(L0_2f[...,0], L0_2f[...,1])
        self.warp_right_0 = build_vpi_warp(R0_2f[...,0], R0_2f[...,1])
        self.warp_left_1  = build_vpi_warp(L1_2f[...,0], L1_2f[...,1])
        self.warp_right_1 = build_vpi_warp(R1_2f[...,0], R1_2f[...,1])

        self.bridge = CvBridge()
        self.raw_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.rect_pubs = [
            self.create_publisher(Image, f'/camera{i}/rectified', 10)
            for i in range(4)
        ]
        self.depth_pubs = [
            self.create_publisher(Image, f'/camera{i}/depth_map', 10)
            for i in range(2)
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

            rectified_rois = []
            
            # All heavy work on GPU
            with vpi.Backend.CUDA:
                # NV12 -> BGR8 on CUDA (queued on our stream)
                vpi_bgr  = vpi_host.convert(vpi.Format.BGR8, stream=self.vpi_stream)

                # 4 even-sized quadrants
                half_w = (w // 2) & ~1
                half_h = (h // 2) & ~1

                rects = (
                    vpi.RectangleI(0,       0,       half_w, half_h),  # 0: TL
                    vpi.RectangleI(half_w,  0,       half_w, half_h),  # 1: TR
                    vpi.RectangleI(0,       half_h,  half_w, half_h),  # 2: BL
                    vpi.RectangleI(half_w,  half_h,  half_w, half_h),  # 3: BR
                )
                rois = [vpi.Image.view(vpi_bgr, r) for r in rects]

                # ---- Rectify each ROI on CUDA using prebuilt warp maps ----
                rectified_0L = rois[1].remap(self.warp_left_0,  interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO, stream=self.vpi_stream)
                rectified_0R = rois[0].remap(self.warp_right_0, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO, stream=self.vpi_stream)
                rectified_1L = rois[3].remap(self.warp_left_1,  interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO, stream=self.vpi_stream)
                rectified_1R = rois[2].remap(self.warp_right_1, interp=vpi.Interp.LINEAR, border=vpi.Border.ZERO, stream=self.vpi_stream)

                rectified_rois = [rectified_0R, rectified_0L, rectified_1R, rectified_1L]
                
                # Stereo depth (pair 0 = [0R,0L], pair1 = [1R,1L])
                depth_maps = []
                for (left_img,right_img) in [(rectified_rois[1],rectified_rois[0]), (rectified_rois[3],rectified_rois[2])]:
                    left_gray  = left_img.convert(vpi.Format.Y16_ER)
                    right_gray = right_img.convert(vpi.Format.Y16_ER)
                    disp = vpi.stereodisp(
                        left_gray,
                        right_gray,
                        backend=vpi.Backend.CUDA,
                        maxdisp=128,
                        mindisp=0,
                        window=3,
                        uniqueness=0.6,
                        includediagonals = False,
                    )
                    disp_f32 = disp.convert(vpi.Format.F32).cpu()/32.0
                    depth = np.zeros_like(disp_f32, dtype=np.float32)
                    valid = disp_f32 > 0
                    depth[valid] = (self.fx * self.baseline_m) / disp_f32[valid]
                    depth[depth < 0.02] = 0.0     # too close, usually invalid
                    depth_maps.append(depth)

            # Sync once so all CUDA ops finish before CPU downloads
            self.vpi_stream.sync()
            
            # Download raw full frame ONCE
            raw_full = vpi_bgr.cpu()
            now = self.get_clock().now().to_msg()

            # Publish the full raw frame
            msg_raw = self.bridge.cv2_to_imgmsg(raw_full, encoding='bgr8')
            msg_raw.header.stamp = now
            msg_raw.header.frame_id = "camera"
            self.raw_pub.publish(msg_raw)

            # Publish rectified quadrants
            for i, rimg in enumerate(rectified_rois):
                bgr_np = rimg.cpu()
                msg = self.bridge.cv2_to_imgmsg(bgr_np, encoding='bgr8')
                msg.header.stamp = now
                msg.header.frame_id = f"camera{i}"
                self.rect_pubs[i].publish(msg)
                
            # Publish depth maps (two stereo pairs)
            for i, depth in enumerate(depth_maps):
                depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
                depth_msg.header.stamp = now
                depth_msg.header.frame_id = f"depth{i}"
                self.depth_pubs[i].publish(depth_msg)

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
