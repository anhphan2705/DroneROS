#!/usr/bin/env python3
import os
import signal
import subprocess
import threading
import time
import logging
from typing import Dict, Optional

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.subscription import Subscription
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

# ── CONFIG ─────────────────────────────────────────────────────────────────────
class Settings:
    ros_workspace: str = os.path.expanduser('~/BaseROS/ros2_ws')
    default_topics = [
        '/camera/image_raw',         # first default feed
        '/camera/rectified_0/right',  # second default feed
        '/camera/rectified_0/left',  # second default feed
    ]
    mjpeg_fps: int = 15               # lowered FPS for stability
    downscale: float = 1.00           # downscale factor for performance
    jpeg_quality: int = 100           # JPEG quality (0-100)
    api_prefix: str = '/api'
    static_prefix: str = '/static'

    # convenience: the first topic to use when no query param is given
    default_topic: str = default_topics[0]

settings = Settings()

# ── LOGGER ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger('hmi_app')

# ── FASTAPI APP ────────────────────────────────────────────────────────────────
app = FastAPI(
    title='Seeker HMI',
    description='Web-based HMI for monitoring and control of drone base station',
    version='0.4'  # bumped version
)

# Serve static files under /static
pkg_share = get_package_share_directory('hmi')
static_dir = os.path.join(pkg_share, 'static')
app.mount(settings.static_prefix, StaticFiles(directory=static_dir), name='static')

@app.get('/', response_class=FileResponse)
def index():
    return os.path.join(static_dir, 'index.html')

# ── ROS2 SETUP ──────────────────────────────────────────────────────────────────
processes: Dict[str, subprocess.Popen] = {}
ros_node: Optional[Node] = None
bridge = CvBridge()

# store latest raw frames per topic (CV2 images)
topic_frames: Dict[str, Optional[np.ndarray]] = {}
frame_lock = threading.Lock()

# keep track of active subscriptions
topic_subs: Dict[str, Subscription] = {}

# Data models
class TopicList(BaseModel):
    topics: list[str]

class Topic(BaseModel):
    name: str

class Job(BaseModel):
    name: str

# ROS spin loop
def ros_spin():
    rclpy.spin(ros_node)

@app.on_event('startup')
def on_startup():
    global ros_node
    rclpy.init()
    ros_node = Node('hmi_api_node')

    # Subscribe to each default topic
    for t in settings.default_topics:
        topic_frames[t] = None
        sub = ros_node.create_subscription(
            RosImage, t,
            lambda msg, tn=t: _handle_image(tn, msg),
            qos_profile=10
        )
        topic_subs[t] = sub
        logger.info(f'Initially subscribed to default topic: {t}')

    # Start ROS spin thread
    threading.Thread(target=ros_spin, daemon=True).start()

@app.on_event('shutdown')
def on_shutdown():
    logger.info('Shutting down HMI server…')
    # first kill any launched missions
    for proc in processes.values():
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            logger.info(f'Sent SIGINT to mission (pid {proc.pid})')
    # then shut down ROS client and server
    if ros_node:
        ros_node.destroy_node()
        rclpy.shutdown()

# Image callback stores raw CV frame only
def _handle_image(topic: str, msg: RosImage):
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        logger.debug(f"Skipping topic {topic} (encoding={msg.encoding}): {e}")
        return
    with frame_lock:
        topic_frames[topic] = cv_img

# ── API ENDPOINTS ───────────────────────────────────────────────────────────────
@app.get(settings.api_prefix + '/topics', response_model=TopicList)
def list_topics():
    """Return all currently subscribed image topics."""
    return TopicList(topics=list(topic_subs.keys()))

@app.post(settings.api_prefix + '/subscribe')
def subscribe_topic(topic: Topic):
    """Start feeding frames from this Image topic."""
    t = topic.name
    if t in topic_subs:
        return JSONResponse({'status': 'already subscribed'})
    topic_frames[t] = None
    sub = ros_node.create_subscription(
        RosImage, t,
        lambda msg, tn=t: _handle_image(tn, msg),
        qos_profile=10
    )
    topic_subs[t] = sub
    logger.info(f'Subscribed to topic: {t}')
    return {'status': 'subscribed'}

@app.post(settings.api_prefix + '/unsubscribe')
def unsubscribe_topic(topic: Topic):
    """Stop feeding frames from this Image topic."""
    t = topic.name
    sub = topic_subs.pop(t, None)
    if sub is None:
        return JSONResponse({'status': 'not subscribed'}, status_code=400)
    ros_node.destroy_subscription(sub)
    topic_frames.pop(t, None)
    logger.info(f'Unsubscribed from topic: {t}')
    return {'status': 'unsubscribed'}

@app.get(settings.api_prefix + '/status')
def status():
    """Return running status of launched jobs."""
    return {name: (proc.poll() is None) for name, proc in processes.items()}

@app.post(settings.api_prefix + '/launch')
def launch(job: Job):
    """Launch a ROS2 launch file by job name."""
    if job.name in processes and processes[job.name].poll() is None:
        return JSONResponse(status_code=400, content={'error': 'Already running'})
    cmd = [
        'bash', '-lc',
        f"source /opt/ros/humble/setup.bash && source {settings.ros_workspace}/install/setup.bash && ros2 launch perception camera_launch.py"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    processes[job.name] = proc
    logger.info(f'Launched {job.name} (pid {proc.pid})')
    return {'status': 'launched'}

@app.post(settings.api_prefix + '/stop')
def stop(job: Job):
    """Stop a launched job by name."""
    proc = processes.get(job.name)
    if not proc or proc.poll() is not None:
        return JSONResponse(status_code=400, content={'error': 'Not running'})
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    logger.info(f'Stopped {job.name}')
    return {'status': 'stopped'}

@app.get('/video_feed')
def video_feed(topic: str = Query(settings.default_topic)):
    """
    Stream MJPEG frames for the given image topic.
    Example: /video_feed?topic=/camera/image_raw
    """
    if topic not in topic_frames:
        raise HTTPException(status_code=404, detail='Topic not found')
    boundary = b'--frame'
    def generator():
        while True:
            with frame_lock:
                frame = topic_frames.get(topic)
            if frame is not None:
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (int(w*settings.downscale), int(h*settings.downscale)))
                ret, jpeg = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality])
                if not ret:
                    continue
                data = jpeg.tobytes()
                header = (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(data)}\r\n\r\n".encode()
                )
                yield header + data + b"\r\n"
            time.sleep(1.0 / settings.mjpeg_fps)
    return StreamingResponse(generator(), media_type='multipart/x-mixed-replace; boundary=frame')

# ── ENTRY POINT ────────────────────────────────────────────────────────────────
def main():
    import uvicorn
    try:
        uvicorn.run(app, host='0.0.0.0', port=8080)
    finally:
        for proc in processes.values():
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)

if __name__ == '__main__':
    main()