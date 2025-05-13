#!/usr/bin/env python3
import os
import signal
import subprocess
import threading
import time
from typing import Dict, Optional

import cv2
import uvicorn
import rclpy
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VIDEO_TOPIC = '/camera/image_raw'    # change to your actual ROS2 image topic
MJPEG_FPS   = 15                     # how many frames per second to serve

# ── FASTAPI APP ────────────────────────────────────────────────────────────────
app = FastAPI()
static_dir = os.path.join(
    get_package_share_directory('hmi'),
    'static'
)
# 1) serve all files in static_dir under /static/*
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 2) serve index.html at GET /
@app.get("/")
def index():
    return FileResponse(os.path.join(static_dir, "index.html"))

# ── ROS2 SETUP ──────────────────────────────────────────────────────────────────
processes: Dict[str, subprocess.Popen] = {}
ros_node: Optional[Node] = None
bridge = CvBridge()
_latest_frame = None
_frame_lock = threading.Lock()

def image_callback(msg: RosImage):
    """ROS2 subscriber callback: store the latest frame."""
    global _latest_frame
    # convert to OpenCV image
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    with _frame_lock:
        _latest_frame = cv_img

def ros_spin():
    rclpy.spin(ros_node)

@app.on_event("startup")
def startup_event():
    global ros_node
    rclpy.init()
    ros_node = Node('hmi_api_node')

    # subscribe to video stream
    ros_node.create_subscription(
        RosImage,
        VIDEO_TOPIC,
        image_callback,
        10  # QoS depth
    )

    # spin ROS2 in background
    threading.Thread(target=ros_spin, daemon=True).start()

@app.on_event("shutdown")
def shutdown_event():
    if ros_node:
        ros_node.destroy_node()
        rclpy.shutdown()

# ── JOB MODEL ──────────────────────────────────────────────────────────────────
class Job(BaseModel):
    name: str

# ── CONTROL ENDPOINTS ──────────────────────────────────────────────────────────
@app.post("/api/launch")
def launch(job: Job):
    if job.name in processes and processes[job.name].poll() is None:
        raise HTTPException(400, f"'{job.name}' already running")
    cmd = [
        "bash", "-lc",
        "source ~/BaseROS/ros2_ws/install/setup.bash && ros2 launch perception camera_launch.py"
    ]
    proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
    processes[job.name] = proc
    return {"status": "launched", "pid": proc.pid}

@app.post("/api/stop")
def stop(job: Job):
    proc = processes.get(job.name)
    if not proc or proc.poll() is not None:
        raise HTTPException(400, f"'{job.name}' not running")
    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    return {"status": "stopped"}

@app.get("/api/status")
def status():
    return {name: (p.poll() is None) for name, p in processes.items()}

# ── VIDEO STREAM ENDPOINT ──────────────────────────────────────────────────────
def mjpeg_generator():
    """Yields multipart JPEG frames for MJPEG streaming."""
    boundary = b"--frame"
    while True:
        with _frame_lock:
            frame = _latest_frame.copy() if _latest_frame is not None else None

        if frame is not None:
            # encode as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                time.sleep(1.0 / MJPEG_FPS)
                continue
            chunk = jpeg.tobytes()

            # build a pure-bytes header
            header = (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(chunk)}\r\n\r\n".encode()
            )

            yield header + chunk + b"\r\n"

        time.sleep(1.0 / MJPEG_FPS)

@app.get("/video_feed")
def video_feed():
    """MJPEG stream of the latest ROS2 camera topic."""
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# ── ENTRY POINT ────────────────────────────────────────────────────────────────
def main():
    uvicorn.run("hmi.app:app", host="0.0.0.0", port=8080, log_level="info")

if __name__ == "__main__":
    main()
