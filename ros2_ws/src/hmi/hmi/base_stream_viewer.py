#!/usr/bin/env python3
import argparse, json, socket, threading, time
import numpy as np
import cv2
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

# ----------------- JSON listener -----------------
class JsonListener(threading.Thread):
    def __init__(self, port, source_id, max_age=0.3):
        super().__init__(daemon=True)
        self.port = port
        self.source_id = source_id
        self.max_age = max_age
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", port))
        self._latest = None
        self._ts = 0
        self._stop = False

    def run(self):
        while not self._stop:
            try:
                data, _ = self.sock.recvfrom(8192)
                obj = json.loads(data.decode("utf-8", errors="ignore"))
                if obj.get("source_id") == self.source_id:
                    self._latest = obj
                    self._ts = time.time()
            except Exception:
                pass

    def stop(self):
        self._stop = True
        self.sock.close()

    def get_latest(self):
        if self._latest is None:
            return None
        if time.time() - self._ts > self.max_age:
            return None
        return self._latest

# ----------------- Drawing -----------------
def draw_boxes(frame, meta, color=(0,128,255)):
    if meta is None:
        return
    for b in meta.get("boxes", []):
        x1, y1, x2, y2 = b["x_min"], b["y_min"], b["x_max"], b["y_max"]
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        label = f"tID:{b['track_id']} {b['class_name']}:{b['class_id']}"
        if b.get("depth", -1) > 0:
            label += f" {b['depth']:.1f}m"
        if b.get("speed_mps", -1) > 0:
            label += f" {b['speed_mps']:.1f}m/s"

        y_text = max(0, y1-5)
        cv2.putText(frame, label, (x1,y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, label, (x1,y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1, cv2.LINE_AA)

# ----------------- GStreamer -----------------
class GStreamerClient:
    def __init__(self, port):
        Gst.init(None)
        pipe_str = (
            f"udpsrc port={port} caps=application/x-rtp,media=video,encoding-name=H264,payload=96 ! "
            "rtpjitterbuffer latency=0 do-lost=true ! "
            "rtph264depay ! h264parse ! "
            "nvv4l2decoder enable-max-performance=1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        self.pipeline = Gst.parse_launch(pipe_str)
        self.appsink = self.pipeline.get_by_name("sink")
        self.appsink.connect("new-sample", self.on_sample)
        self.frame = None
        self.pipeline.set_state(Gst.State.PLAYING)

    def on_sample(self, sink):
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value("width")
        h = caps.get_structure(0).get_value("height")
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        arr = np.frombuffer(mapinfo.data, np.uint8)
        # Avoid extra copy: only copy once for overlay safety
        if arr.size >= w*h*3:
            self.frame = arr[:w*h*3].reshape((h,w,3)).copy()
        buf.unmap(mapinfo)
        return Gst.FlowReturn.OK

    def get_frame(self):
        return self.frame

# ----------------- Main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-port", type=int, default=5600)
    ap.add_argument("--json0-port", type=int, default=6000)
    ap.add_argument("--json1-port", type=int, default=6001)
    args = ap.parse_args()

    j0 = JsonListener(args.json0_port, source_id=0); j0.start()
    j1 = JsonListener(args.json1_port, source_id=2); j1.start()

    gst = GStreamerClient(args.video_port)

    print("Running overlay client (1920x1080, HW decode). Press q to quit.")
    
    try:
        cv2.namedWindow("Overlay Stream", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Overlay Stream", 960, 540)

        while True:
            frame = gst.get_frame()
            if frame is None:
                time.sleep(0.002)
                continue

            h,w = frame.shape[:2]
            q_w, q_h = w//2, h//2

            # Left quadrants
            tl = frame[0:q_h, 0:q_w]
            bl = frame[q_h:h, 0:q_w]

            draw_boxes(tl, j0.get_latest())
            draw_boxes(bl, j1.get_latest())

            cv2.imshow("Overlay Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        j0.stop(); j1.stop()
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
    
# chmod +x overlay_client.py
# ./base_stream_viewer.py --video-port 5600 --json0-port 6000 --json1-port 6001
# or 
# python3 overlay_client.py --video-port 5600 --json0-port 6000 --json1-port 6001 | cv2.namedWindow("Overlay 1080p", cv2.WINDOW_NORMAL)
