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
        '/camera/image_raw',
        '/camera/rectified_0/right',
        '/camera/rectified_0/left',
    ]
    mjpeg_fps: int = 15
    downscale: float = 1.00
    jpeg_quality: int = 100
    api_prefix: str = '/api'
    static_prefix: str = '/static'
    default_topic: str = default_topics[0]

settings = Settings()

# ── LOGGER ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger('hmi_app')

# ── FASTAPI APP ────────────────────────────────────────────────────────────────
app = FastAPI(
    title='Seeker HMI',
    description='Web-based HMI for monitoring and control of drone base station',
    version='0.4'
)

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

# store latest raw frames per topic
topic_frames: Dict[str, Optional[np.ndarray]] = {}
# store the ROS time (in seconds) when the last frame arrived
last_frame_time: Dict[str, float] = {}
frame_lock = threading.Lock()

topic_subs: Dict[str, Subscription] = {}

class TopicList(BaseModel):
    topics: list[str]

class Topic(BaseModel):
    name: str

class Job(BaseModel):
    name: str

def ros_spin():
    rclpy.spin(ros_node)

@app.on_event('startup')
def on_startup():
    global ros_node
    rclpy.init()
    ros_node = Node('hmi_api_node')

    for t in settings.default_topics:
        topic_frames[t] = None
        # initialize with zero (meaning “no frames yet”)
        last_frame_time[t] = 0.0
        sub = ros_node.create_subscription(
            RosImage, t,
            lambda msg, tn=t: _handle_image(tn, msg),
            qos_profile=10
        )
        topic_subs[t] = sub
        logger.info(f'Initially subscribed to default topic: {t}')

    threading.Thread(target=ros_spin, daemon=True).start()

@app.on_event('shutdown')
def on_shutdown():
    logger.info('Shutting down HMI server…')
    for proc in processes.values():
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    if ros_node:
        ros_node.destroy_node()
        rclpy.shutdown()

def _handle_image(topic: str, msg: RosImage):
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        logger.debug(f"Skipping topic {topic} (encoding={msg.encoding}): {e}")
        return

    with frame_lock:
        topic_frames[topic] = cv_img
        # record ROS time (seconds since ROS epoch/current sim time)
        last_frame_time[topic] = ros_node.get_clock().now().nanoseconds / 1e9

# ── API ENDPOINTS ───────────────────────────────────────────────────────────────
@app.get(settings.api_prefix + '/topics', response_model=TopicList)
def list_topics():
    return TopicList(topics=list(topic_subs.keys()))

@app.get(settings.api_prefix + '/last_frame')
def last_frame(topic: str = Query(..., description="Topic name")):
    """
    Return the ROS timestamp (in seconds) when the last frame was received.
    """
    ts = last_frame_time.get(topic)
    if ts is None or ts == 0.0:
        raise HTTPException(status_code=404, detail='No frames received yet for this topic')
    return {'last_frame': ts}

@app.post(settings.api_prefix + '/subscribe')
def subscribe_topic(topic: Topic):
    t = topic.name
    if t in topic_subs:
        return JSONResponse({'status': 'already subscribed'})
    topic_frames[t] = None
    last_frame_time[t] = 0.0
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
    t = topic.name
    sub = topic_subs.pop(t, None)
    if sub is None:
        return JSONResponse({'status': 'not subscribed'}, status_code=400)
    ros_node.destroy_subscription(sub)
    topic_frames.pop(t, None)
    last_frame_time.pop(t, None)
    logger.info(f'Unsubscribed from topic: {t}')
    return {'status': 'unsubscribed'}

@app.get(settings.api_prefix + '/status')
def status():
    return {name: (proc.poll() is None) for name, proc in processes.items()}

@app.post(settings.api_prefix + '/launch')
def launch(job: Job):
    if job.name in processes and processes[job.name].poll() is None:
        return JSONResponse(status_code=400, content={'error': 'Already running'})
    cmd = [
        'bash', '-lc',
        f"source /opt/ros/humble/setup.bash && source {settings.ros_workspace}/install/setup.bash && ros2 launch perception camera_launch.py"
    ]
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            preexec_fn=os.setsid)
    processes[job.name] = proc
    logger.info(f'Launched {job.name} (pid {proc.pid})')
    return {'status': 'launched'}

@app.post(settings.api_prefix + '/stop')
def stop(job: Job):
    proc = processes.get(job.name)
    if not proc or proc.poll() is not None:
        return JSONResponse(status_code=400, content={'error': 'Not running'})
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    logger.info(f'Stopped {job.name}')
    return {'status': 'stopped'}

@app.get('/video_feed')
def video_feed(topic: str = Query(settings.default_topic)):
    if topic not in topic_frames:
        raise HTTPException(status_code=404, detail='Topic not found')
    boundary = b'--frame'
    def generator():
        while True:
            with frame_lock:
                frame = topic_frames.get(topic)
            if frame is not None:
                h, w = frame.shape[:2]
                small = cv2.resize(frame,
                                   (int(w*settings.downscale), int(h*settings.downscale)))
                ret, jpeg = cv2.imencode('.jpg', small,
                                         [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality])
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
    return StreamingResponse(generator(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

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