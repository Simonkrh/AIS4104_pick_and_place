#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose


def _stamp_to_nanoseconds(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _normalize_quaternion_xyzw(
    x: float, y: float, z: float, w: float
) -> tuple[float, float, float, float]:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


def _rotate_vector_by_quaternion_xyzw(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
    vx: float,
    vy: float,
    vz: float,
) -> tuple[float, float, float]:
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    vpx = vx + qw * tx + (qy * tz - qz * ty)
    vpy = vy + qw * ty + (qz * tx - qx * tz)
    vpz = vz + qw * tz + (qx * ty - qy * tx)
    return vpx, vpy, vpz


class Detection3DTransformNode(Node):
    def __init__(self):
        super().__init__("detection_3d_transform_node")

        def parse_bool(value) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            text = str(value).strip().lower()
            if text in ("1", "true", "yes", "on"):
                return True
            if text in ("0", "false", "no", "off", ""):
                return False
            return bool(text)

        self.declare_parameter("input_topic", "/yolo/detections_3d")
        self.declare_parameter("output_topic", "/yolo/detections_3d_base")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("tf_timeout_sec", 0.05)
        self.declare_parameter("allow_latest_tf_fallback", True)
        self.declare_parameter("max_input_stamp_age_sec", 2.0)

        self.declare_parameter("target_class", "")
        self.declare_parameter("min_score", 0.0)
        self.declare_parameter("best_pose_topic", "/pick_target_pose")
        self.declare_parameter("best_tf_child_frame", "detected_object")

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.allow_latest_tf_fallback = parse_bool(
            self.get_parameter("allow_latest_tf_fallback").value
        )
        self.max_input_stamp_age_sec = float(
            self.get_parameter("max_input_stamp_age_sec").value
        )

        self.target_class = str(self.get_parameter("target_class").value).strip()
        self.min_score = float(self.get_parameter("min_score").value)
        self.best_pose_topic = str(self.get_parameter("best_pose_topic").value).strip()
        self.best_tf_child_frame = str(
            self.get_parameter("best_tf_child_frame").value
        ).strip()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = (
            TransformBroadcaster(self) if self.best_tf_child_frame else None
        )

        self.pub = self.create_publisher(Detection3DArray, self.output_topic, 10)
        self.best_pose_pub = (
            self.create_publisher(PoseStamped, self.best_pose_topic, 10)
            if self.best_pose_topic
            else None
        )

        self.sub = self.create_subscription(
            Detection3DArray, self.input_topic, self.on_detections, 10
        )

        self._last_tf_warn_ns = 0
        self._last_stamp_warn_ns = 0

        self.get_logger().info(f"Subscribing detections: {self.input_topic}")
        self.get_logger().info(
            f"Publishing transformed detections: {self.output_topic}"
        )
        self.get_logger().info(f"Target frame: {self.target_frame}")
        if self.best_pose_pub is not None:
            self.get_logger().info(f"Publishing best pose: {self.best_pose_topic}")
        if self.tf_broadcaster is not None:
            self.get_logger().info(
                f"Broadcasting TF for best detection: "
                f"{self.target_frame} -> {self.best_tf_child_frame}"
            )

    def _is_stamp_stale(self, stamp) -> bool:
        stamp_ns = _stamp_to_nanoseconds(stamp)
        if stamp_ns <= 0:
            return True
        now_ns = self.get_clock().now().nanoseconds
        max_age_ns = int(self.max_input_stamp_age_sec * 1e9)
        return abs(now_ns - stamp_ns) > max_age_ns

    def _warn_stale_stamp_once(self, stamp) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_stamp_warn_ns > 1_000_000_000:
            self._last_stamp_warn_ns = now_ns
            age_sec = (now_ns - _stamp_to_nanoseconds(stamp)) / 1e9
            self.get_logger().warn(
                f"Incoming detection timestamp is stale by {age_sec:.3f}s; "
                "using latest TF and current node time for outputs."
            )

    def _lookup_target_t_source(
        self,
        source_frame: str,
        stamp,
        force_latest: bool = False,
    ) -> Optional[TransformStamped]:
        exact_lookup_error = ""
        timeout = Duration(seconds=max(self.tf_timeout_sec, 0.0))

        if force_latest:
            query_times = [Time()]
        else:
            query_times = [Time.from_msg(stamp)]
            if self.allow_latest_tf_fallback:
                query_times.append(Time())

        for query_time in query_times:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    query_time,
                    timeout=timeout,
                )
                if query_time.nanoseconds == 0 and exact_lookup_error:
                    now_ns = self.get_clock().now().nanoseconds
                    if now_ns - self._last_tf_warn_ns > 1_000_000_000:
                        self._last_tf_warn_ns = now_ns
                        self.get_logger().warn(
                            "TF lookup fell back to latest transform instead of "
                            f"message timestamp. Exact lookup error: {exact_lookup_error}"
                        )
                return transform
            except TransformException as exc:
                if query_time.nanoseconds != 0 and not exact_lookup_error:
                    exact_lookup_error = str(exc)
                continue

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_tf_warn_ns > 1_000_000_000:
            self._last_tf_warn_ns = now_ns
            self.get_logger().warn(
                f"Could not resolve TF {self.target_frame} <- {source_frame} "
                f"(allow_latest_tf_fallback={self.allow_latest_tf_fallback}). "
                f"Last error: {exact_lookup_error}"
            )
        return None

    @staticmethod
    def _transform_point(
        transform: TransformStamped, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        q = transform.transform.rotation
        t = transform.transform.translation
        qx, qy, qz, qw = _normalize_quaternion_xyzw(
            float(q.x), float(q.y), float(q.z), float(q.w)
        )
        rx, ry, rz = _rotate_vector_by_quaternion_xyzw(qx, qy, qz, qw, x, y, z)
        return rx + float(t.x), ry + float(t.y), rz + float(t.z)

    def on_detections(self, msg: Detection3DArray):
        source_frame = str(msg.header.frame_id)
        if not source_frame:
            self.get_logger().warn(
                "Received Detection3DArray with empty header.frame_id"
            )
            return

        input_stamp_stale = self._is_stamp_stale(msg.header.stamp)
        if input_stamp_stale:
            self._warn_stale_stamp_once(msg.header.stamp)

        publish_stamp = (
            self.get_clock().now().to_msg() if input_stamp_stale else msg.header.stamp
        )

        transform = None
        if source_frame != self.target_frame:
            transform = self._lookup_target_t_source(
                source_frame,
                msg.header.stamp,
                force_latest=input_stamp_stale,
            )
            if transform is None:
                return

        out = Detection3DArray()
        out.header.stamp = publish_stamp
        out.header.frame_id = self.target_frame

        for det in msg.detections:
            center = det.bbox.center.position
            x = float(center.x)
            y = float(center.y)
            z = float(center.z)

            if transform is not None:
                x, y, z = self._transform_point(transform, x, y, z)

            det_out = Detection3D()
            det_out.header.stamp = publish_stamp
            det_out.header.frame_id = self.target_frame

            det_out.bbox.center.position.x = x
            det_out.bbox.center.position.y = y
            det_out.bbox.center.position.z = z
            det_out.bbox.center.orientation.w = 1.0
            det_out.bbox.size = det.bbox.size

            for hyp in det.results:
                hyp_out = ObjectHypothesisWithPose()
                hyp_out.hypothesis.class_id = hyp.hypothesis.class_id
                hyp_out.hypothesis.score = hyp.hypothesis.score
                hyp_out.pose.covariance = list(hyp.pose.covariance)
                hyp_out.pose.pose.position.x = x
                hyp_out.pose.pose.position.y = y
                hyp_out.pose.pose.position.z = z
                hyp_out.pose.pose.orientation.w = 1.0
                det_out.results.append(hyp_out)

            out.detections.append(det_out)

        self.pub.publish(out)

        best = self._select_best_detection(out)
        if best is None:
            return

        if self.best_pose_pub is not None:
            best_z = float(best.bbox.center.position.z) + max(
                0.5 * float(best.bbox.size.z), 0.0
            )
            pose = PoseStamped()
            pose.header.stamp = publish_stamp
            pose.header.frame_id = out.header.frame_id
            pose.pose.position = best.bbox.center.position
            pose.pose.position.z = best_z
            pose.pose.orientation.w = 1.0
            self.best_pose_pub.publish(pose)

        if self.tf_broadcaster is not None:
            best_z = float(best.bbox.center.position.z) + max(
                0.5 * float(best.bbox.size.z), 0.0
            )
            tf_msg = TransformStamped()
            tf_msg.header.stamp = publish_stamp
            tf_msg.header.frame_id = out.header.frame_id
            tf_msg.child_frame_id = self.best_tf_child_frame
            tf_msg.transform.translation.x = float(best.bbox.center.position.x)
            tf_msg.transform.translation.y = float(best.bbox.center.position.y)
            tf_msg.transform.translation.z = best_z
            tf_msg.transform.rotation.w = 1.0
            self.tf_broadcaster.sendTransform(tf_msg)

    def _select_best_detection(self, msg: Detection3DArray) -> Optional[Detection3D]:
        best_det = None
        best_score = float("-inf")
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0].hypothesis
            cls = str(hyp.class_id)
            score = float(hyp.score)
            if self.target_class and cls != self.target_class:
                continue
            if score < self.min_score:
                continue
            if best_det is None or score > best_score:
                best_det = det
                best_score = score
        return best_det


def main():
    rclpy.init()
    node = Detection3DTransformNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
