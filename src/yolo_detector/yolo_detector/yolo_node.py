#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import Image as ImageMsg
from ultralytics import YOLO
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")

        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("model", "models/pick_place_best.pt")
        self.declare_parameter("conf", 0.65)
        self.declare_parameter("device", "cpu")

        image_topic = str(self.get_parameter("image_topic").value)
        model_path = str(self.get_parameter("model").value)
        self.conf = float(self.get_parameter("conf").value)
        self.device = str(self.get_parameter("device").value)

        if not Path(model_path).exists():
            self.get_logger().warn(
                f"I could not find the model at {model_path}. Using yolov8n.pt instead."
            )
            model_path = "yolov8n.pt"

        self.bridge = CvBridge()
        self.model = YOLO(model_path)
        self.names = self.model.names

        self.sub = self.create_subscription(Image, image_topic, self.cb, 10)
        self.pub = self.create_publisher(Detection2DArray, "/yolo/detections", 10)
        self.pub_img = self.create_publisher(ImageMsg, "/yolo/image_annotated", 10)

        self.get_logger().info(
            f"YOLO is ready with model {model_path}, confidence {self.conf}, device {self.device}."
        )
        self.get_logger().info(f"Reading images from {image_topic}.")
        self.get_logger().info("Publishing YOLO detections.")
        self.get_logger().info("Publishing the annotated YOLO image.")

    @staticmethod
    def _wrap_half_turn(theta_rad: float) -> float:
        while theta_rad <= -0.5 * math.pi:
            theta_rad += math.pi
        while theta_rad > 0.5 * math.pi:
            theta_rad -= math.pi
        return theta_rad

    @staticmethod
    def _wrap_quarter_turn(theta_rad: float) -> float:
        while theta_rad <= -0.25 * math.pi:
            theta_rad += 0.5 * math.pi
        while theta_rad > 0.25 * math.pi:
            theta_rad -= 0.5 * math.pi
        return theta_rad

    def _orientation_kind(self, class_name: str) -> str:
        lowered_name = class_name.strip().lower()
        if "stick" in lowered_name:
            return "stick"
        if "cube" in lowered_name:
            return "cube"
        return ""

    def _estimate_object_theta(
        self,
        image,
        x1i: int,
        y1i: int,
        x2i: int,
        y2i: int,
        orientation_kind: str,
    ) -> tuple[float, bool, tuple[float, float]]:
        roi = image[y1i:y2i, x1i:x2i]
        if roi.size == 0:
            return 0.0, False, (0.0, 0.0)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (0, 40, 40),
            (179, 255, 255),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        roi_area = float(roi.shape[0] * roi.shape[1])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, False, (0.0, 0.0)

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < 0.05 * roi_area:
            return 0.0, False, (0.0, 0.0)

        moments = cv2.moments(contour)
        if abs(float(moments["m00"])) <= 1e-6:
            return 0.0, False, (0.0, 0.0)

        centroid = (
            float(moments["m10"]) / float(moments["m00"]) + float(x1i),
            float(moments["m01"]) / float(moments["m00"]) + float(y1i),
        )

        rect = cv2.minAreaRect(cv2.convexHull(contour))
        _, (width, height), angle_deg = rect
        short_side = min(float(width), float(height))
        long_side = max(float(width), float(height))
        if short_side <= 1.0:
            return 0.0, False, (0.0, 0.0)

        if orientation_kind == "stick" and long_side / short_side < 1.3:
            return 0.0, False, (0.0, 0.0)

        if float(height) > float(width):
            angle_deg += 90.0

        theta = self._wrap_half_turn(math.radians(float(angle_deg)))
        if orientation_kind == "stick":
            theta = self._wrap_half_turn(theta + 0.5 * math.pi)
        if orientation_kind == "cube":
            theta = self._wrap_quarter_turn(theta)
        return theta, True, centroid

    @staticmethod
    def _draw_orientation_line(
        image,
        center_x: float,
        center_y: float,
        theta_rad: float,
        length_px: float,
        color,
    ) -> None:
        dx = 0.5 * length_px * math.cos(theta_rad)
        dy = 0.5 * length_px * math.sin(theta_rad)
        pt1 = (int(round(center_x - dx)), int(round(center_y - dy)))
        pt2 = (int(round(center_x + dx)), int(round(center_y + dy)))
        cv2.line(image, pt1, pt2, color, 2)

    @staticmethod
    def _draw_pick_center(image, center_x: float, center_y: float) -> None:
        center = (int(round(center_x)), int(round(center_y)))
        cv2.drawMarker(
            image,
            center,
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.circle(image, center, 5, (255, 0, 255), 2)

    def cb(self, msg: Image):
        output_stamp = msg.header.stamp
        frame_id = str(msg.header.frame_id)

        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        annotated = cv_img.copy()
        image_height, image_width = cv_img.shape[:2]

        results = self.model.predict(
            cv_img, conf=self.conf, device=self.device, verbose=False
        )

        det_array = Detection2DArray()
        det_array.header.stamp = output_stamp
        det_array.header.frame_id = frame_id

        r = results[0]
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cls_id = int(b.cls[0].item())
                score = float(b.conf[0].item())

                name = self.names.get(cls_id, str(cls_id))

                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                x1i = max(0, min(x1i, image_width - 1))
                x2i = max(0, min(x2i, image_width))
                y1i = max(0, min(y1i, image_height - 1))
                y2i = max(0, min(y2i, image_height))
                cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)

                bbox_center_x = (x1 + x2) / 2.0
                bbox_center_y = (y1 + y2) / 2.0
                pick_center_x = bbox_center_x
                pick_center_y = bbox_center_y
                theta = 0.0
                orientation_kind = self._orientation_kind(name)
                if orientation_kind:
                    theta, theta_found, mask_centroid = self._estimate_object_theta(
                        cv_img,
                        x1i,
                        y1i,
                        x2i,
                        y2i,
                        orientation_kind,
                    )
                    if theta_found:
                        pick_center_x, pick_center_y = mask_centroid
                        self._draw_orientation_line(
                            annotated,
                            pick_center_x,
                            pick_center_y,
                            theta,
                            max(x2 - x1, y2 - y1),
                            (0, 200, 255),
                        )

                cv2.putText(
                    annotated,
                    f"{name} {score:.2f}",
                    (x1i, max(0, y1i - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                self._draw_pick_center(annotated, pick_center_x, pick_center_y)

                det = Detection2D()
                det.header.stamp = output_stamp
                det.header.frame_id = frame_id

                bbox = BoundingBox2D()
                bbox.center.position.x = pick_center_x
                bbox.center.position.y = pick_center_y
                bbox.center.theta = theta
                bbox.size_x = x2 - x1
                bbox.size_y = y2 - y1
                det.bbox = bbox

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = name
                hyp.hypothesis.score = score
                det.results.append(hyp)

                det_array.detections.append(det)

        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        img_msg.header.stamp = output_stamp
        img_msg.header.frame_id = frame_id
        self.pub_img.publish(img_msg)

        self.pub.publish(det_array)


def main():
    rclpy.init()
    node = YoloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
