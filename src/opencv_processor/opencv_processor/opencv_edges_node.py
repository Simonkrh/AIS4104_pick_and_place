#!/usr/bin/env python3
from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class OpenCVEdgesNode(Node):
    def __init__(self):
        super().__init__("opencv_edges_node")

        self.declare_parameter("image_topic", "/image_raw")
        image_topic = str(self.get_parameter("image_topic").value)

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, image_topic, self.on_image, 10)
        self.pub = self.create_publisher(Image, "/opencv/image_edges", 10)

        self.get_logger().info(f"Reading images from {image_topic}.")
        self.get_logger().info("Publishing edge images.")

    def on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)

            out = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
            out.header = msg.header
            self.pub.publish(out)
        except Exception as exc:
            self.get_logger().error(f"OpenCV could not process this image. {exc}")


def main():
    rclpy.init()
    node = OpenCVEdgesNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
