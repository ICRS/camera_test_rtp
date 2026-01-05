#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

class H264RTPSender(Node):

    def __init__(self):
        super().__init__('h264_rtp_sender')

        # ROS parameters
        self.declare_parameter('topic', '/image_raw/compressed')
        self.declare_parameter('dest_ip', '192.168.1.167')
        self.declare_parameter('dest_port', 5004)
        self.declare_parameter('fps', 30)

        self.topic = self.get_parameter('topic').value
        self.dest_ip = self.get_parameter('dest_ip').value
        self.dest_port = self.get_parameter('dest_port').value
        self.fps = self.get_parameter('fps').value

        self.width = None
        self.height = None
        self.first_frame = True
        self.frame_count = 0

        # ROS subscription
        self.sub = self.create_subscription(
            CompressedImage,
            self.topic,
            self.callback,
            10
        )

        # Initialize GStreamer
        Gst.init(None)
        self.pipeline = None
        self.appsrc = None

        self.get_logger().info(f"Waiting for first frame on {self.topic} to configure pipeline...")

    def init_pipeline(self, width, height):
        """Initialize GStreamer pipeline with detected width/height"""
        pipeline_desc = (
            f"appsrc name=src is-live=true block=false format=time "
            f"caps=video/x-raw,format=I420,framerate={self.fps}/1,width={width},height={height} ! "
            f"x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
            f"rtph264pay pt=96 config-interval=1 ! "
            f"udpsink host={self.dest_ip} port={self.dest_port} sync=false async=false"
        )

        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsrc = self.pipeline.get_by_name('src')
        self.pipeline.set_state(Gst.State.PLAYING)

        self.get_logger().info(f"Sending H.264 RTP to {self.dest_ip}:{self.dest_port}, size={width}x{height}, fps={self.fps}")

    def callback(self, msg: CompressedImage):
        if not msg.data:
            return

        # Decode JPEG → BGR
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            self.get_logger().warn("Failed to decode JPEG frame")
            return

        # Detect width/height on first frame
        if self.first_frame:
            self.height, self.width, _ = frame_bgr.shape
            self.init_pipeline(self.width, self.height)
            self.first_frame = False

        # Convert BGR → I420 (YUV420)
        frame_i420 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
        data = frame_i420.tobytes()

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)

        # Set PTS and duration using frame counter
        buf.pts = int(self.frame_count * Gst.SECOND / self.fps)
        buf.duration = int(Gst.SECOND / self.fps)
        self.frame_count += 1

        # Push buffer to appsrc
        ret = self.appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            self.get_logger().warn(f"Failed to push buffer: {ret}")
        else:
            self.get_logger().info(f"Pushed frame {self.frame_count}: {len(data)} bytes, PTS={buf.pts}")

def main():
    rclpy.init()
    node = H264RTPSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.pipeline:
            node.pipeline.set_state(Gst.State.NULL)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
