#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import cv2
from ultralytics import YOLO

from sensor_msgs.msg import Image
from sensor_msgs.msg import Image as ImageMsg
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('conf', 0.65)
        self.declare_parameter('device', 'cpu')

        image_topic = self.get_parameter('image_topic').value
        model_path = self.get_parameter('model').value
        self.conf = float(self.get_parameter('conf').value)
        self.device = str(self.get_parameter('device').value)

        self.bridge = CvBridge()
        self.model = YOLO(model_path)

        # 👇 Load class names from the YOLO model
        self.names = self.model.names

        self.sub = self.create_subscription(Image, image_topic, self.cb, 10)
        self.pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.pub_img = self.create_publisher(ImageMsg, '/yolo/image_annotated', 10)

        self.get_logger().info(f"YOLO model={model_path}, conf={self.conf}, device={self.device}")
        self.get_logger().info(f"Subscribing to: {image_topic}")
        self.get_logger().info("Publishing detections on: /yolo/detections")
        self.get_logger().info("Publishing annotated image on: /yolo/image_annotated")

    def cb(self, msg: Image):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        annotated = cv_img.copy()

        results = self.model.predict(cv_img, conf=self.conf, device=self.device, verbose=False)
        det_array = Detection2DArray()
        det_array.header = msg.header

        r = results[0]
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cls_id = int(b.cls[0].item())
                score = float(b.conf[0].item())

                # 👇 Convert class ID → real name
                name = self.names.get(cls_id, str(cls_id))

                # Draw rectangle + label with NAME instead of ID
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

                cv2.putText(
                    annotated,
                    f"{name} {score:.2f}",
                    (x1i, max(0, y1i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Build ROS detection message
                det = Detection2D()
                det.header = msg.header

                bbox = BoundingBox2D()
                bbox.center.position.x = (x1 + x2) / 2.0
                bbox.center.position.y = (y1 + y2) / 2.0
                bbox.center.theta = 0.0
                bbox.size_x = (x2 - x1)
                bbox.size_y = (y2 - y1)
                det.bbox = bbox

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = name
                hyp.hypothesis.score = score
                det.results.append(hyp)

                det_array.detections.append(det)

        # Publish annotated image
        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        img_msg.header = msg.header
        self.pub_img.publish(img_msg)

        # Publish detections
        self.pub.publish(det_array)


def main():
    rclpy.init()
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

