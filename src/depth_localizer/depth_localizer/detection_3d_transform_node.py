#!/usr/bin/env python3
from __future__ import annotations

import math
from collections import deque
from statistics import median
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, String
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


def _quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


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

        self.declare_parameter("input_topic", "/yolo/detections_3d")
        self.declare_parameter("output_topic", "/yolo/detections_3d_base")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("tf_timeout_sec", 0.05)
        self.declare_parameter("allow_latest_tf_fallback", True)
        self.declare_parameter("max_input_stamp_age_sec", 2.0)

        self.declare_parameter("target_class", "")
        self.declare_parameter("min_score", 0.0)
        self.declare_parameter("min_transformed_z", -0.05)
        self.declare_parameter("max_transformed_z", 0.30)
        self.declare_parameter("best_pose_topic", "/pick_target_pose")
        self.declare_parameter("best_class_topic", "/pick_target_class")
        self.declare_parameter("best_tf_child_frame", "detected_object")
        self.declare_parameter("motion_active_topic", "/pick_motion_active")
        self.declare_parameter("best_pose_filter_window_size", 1)
        self.declare_parameter("best_pose_jump_rejection_distance", 0.03)
        self.declare_parameter("best_pose_jump_rejection_hold_sec", 0.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.tf_timeout_sec = float(self.get_parameter("tf_timeout_sec").value)
        self.allow_latest_tf_fallback = bool(
            self.get_parameter("allow_latest_tf_fallback").value
        )
        self.max_input_stamp_age_sec = float(
            self.get_parameter("max_input_stamp_age_sec").value
        )

        self.target_class = str(self.get_parameter("target_class").value).strip()
        self.min_score = float(self.get_parameter("min_score").value)
        self.min_transformed_z = float(self.get_parameter("min_transformed_z").value)
        self.max_transformed_z = float(self.get_parameter("max_transformed_z").value)
        self.best_pose_topic = str(self.get_parameter("best_pose_topic").value).strip()
        self.best_class_topic = str(
            self.get_parameter("best_class_topic").value
        ).strip()
        self.best_tf_child_frame = str(
            self.get_parameter("best_tf_child_frame").value
        ).strip()
        self.motion_active_topic = str(
            self.get_parameter("motion_active_topic").value
        ).strip()
        self.best_pose_filter_window_size = max(
            int(self.get_parameter("best_pose_filter_window_size").value), 1
        )
        self.best_pose_jump_rejection_distance = max(
            float(self.get_parameter("best_pose_jump_rejection_distance").value), 0.0
        )
        self.best_pose_jump_rejection_hold_sec = max(
            float(self.get_parameter("best_pose_jump_rejection_hold_sec").value), 0.0
        )

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
        self.best_class_pub = (
            self.create_publisher(String, self.best_class_topic, 10)
            if self.best_class_topic
            else None
        )

        self.sub = self.create_subscription(
            Detection3DArray, self.input_topic, self.on_detections, 10
        )
        self.motion_active = False
        self.motion_active_sub = self.create_subscription(
            Bool, self.motion_active_topic, self._on_motion_active, 10
        )

        self._last_tf_warn_ns = 0
        self._last_stamp_warn_ns = 0
        self._accepted_best_position: Optional[tuple[float, float, float]] = None
        self._pending_best_position: Optional[tuple[float, float, float]] = None
        self._pending_best_since_ns: Optional[int] = None
        self._best_position_samples = deque(maxlen=self.best_pose_filter_window_size)

        self.get_logger().info(f"Reading detections from {self.input_topic}.")
        self.get_logger().info(
            f"Publishing transformed detections on {self.output_topic}."
        )
        self.get_logger().info(f"Using target frame {self.target_frame}.")
        if self.best_pose_pub is not None:
            self.get_logger().info(f"Publishing the best pose on {self.best_pose_topic}.")
        if self.best_class_pub is not None:
            self.get_logger().info(f"Publishing the best class on {self.best_class_topic}.")
        if self.tf_broadcaster is not None:
            self.get_logger().info(
                f"Broadcasting TF for the best detection from "
                f"{self.target_frame} to {self.best_tf_child_frame}."
            )
        self.get_logger().info(
            f"Reading the motion active flag from {self.motion_active_topic}."
        )
        self.get_logger().info(
            "Best pose smoothing is on. "
            f"Window is {self.best_pose_filter_window_size} samples."
        )
        self.get_logger().info(
            "Best pose jump rejection is on. "
            f"Distance is {self.best_pose_jump_rejection_distance:.3f} m, "
            f"hold time is {self.best_pose_jump_rejection_hold_sec:.2f} s."
        )

    def _on_motion_active(self, msg: Bool) -> None:
        new_motion_active = bool(msg.data)
        if new_motion_active != self.motion_active:
            self._reset_best_pose_filters()
        self.motion_active = new_motion_active

    def _reset_best_pose_filters(self) -> None:
        self._accepted_best_position = None
        self._pending_best_position = None
        self._pending_best_since_ns = None
        self._best_position_samples.clear()

    @staticmethod
    def _distance_between_points(
        a: tuple[float, float, float], b: tuple[float, float, float]
    ) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        dz = a[2] - b[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _stabilize_best_position(
        self, position: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        if self._accepted_best_position is None:
            self._accepted_best_position = position
            self._pending_best_position = None
            self._pending_best_since_ns = None
            return position

        if (
            self._distance_between_points(position, self._accepted_best_position)
            <= self.best_pose_jump_rejection_distance
        ):
            self._pending_best_position = None
            self._pending_best_since_ns = None
            self._accepted_best_position = position
            return position

        if self.best_pose_jump_rejection_hold_sec <= 0.0:
            self._accepted_best_position = position
            self._pending_best_position = None
            self._pending_best_since_ns = None
            return position

        now_ns = self.get_clock().now().nanoseconds

        if (
            self._pending_best_position is None
            or self._distance_between_points(position, self._pending_best_position)
            > self.best_pose_jump_rejection_distance
        ):
            self._pending_best_position = position
            self._pending_best_since_ns = now_ns
            return self._accepted_best_position

        if self._pending_best_since_ns is None:
            self._pending_best_since_ns = now_ns
            return self._accepted_best_position

        hold_ns = int(self.best_pose_jump_rejection_hold_sec * 1e9)
        if now_ns - self._pending_best_since_ns < hold_ns:
            return self._accepted_best_position

        self._accepted_best_position = position
        self._pending_best_position = None
        self._pending_best_since_ns = None
        return position

    def _filter_best_position(
        self, position: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        self._best_position_samples.append(position)
        if len(self._best_position_samples) <= 1:
            return position

        return (
            float(median(sample[0] for sample in self._best_position_samples)),
            float(median(sample[1] for sample in self._best_position_samples)),
            float(median(sample[2] for sample in self._best_position_samples)),
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
                f"The incoming detection is {age_sec:.3f} seconds old. "
                "Using the latest TF and the current node time."
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
                            "TF lookup used the latest transform instead of the message time. "
                            f"The exact lookup error was {exact_lookup_error}."
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
                f"I could not find TF from {source_frame} to {self.target_frame}. "
                f"Latest TF fallback is {self.allow_latest_tf_fallback}. "
                f"Last error was {exact_lookup_error}."
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

    @staticmethod
    def _transform_in_plane_orientation_delta(
        transform: Optional[TransformStamped], orientation
    ) -> tuple[float, float, float, float]:
        source_qx, source_qy, source_qz, source_qw = _normalize_quaternion_xyzw(
            float(orientation.x),
            float(orientation.y),
            float(orientation.z),
            float(orientation.w),
        )

        if transform is None:
            tf_qx, tf_qy, tf_qz, tf_qw = 0.0, 0.0, 0.0, 1.0
        else:
            tf_rotation = transform.transform.rotation
            tf_qx, tf_qy, tf_qz, tf_qw = _normalize_quaternion_xyzw(
                float(tf_rotation.x),
                float(tf_rotation.y),
                float(tf_rotation.z),
                float(tf_rotation.w),
            )

        baseline_x, baseline_y, _ = _rotate_vector_by_quaternion_xyzw(
            tf_qx, tf_qy, tf_qz, tf_qw, 1.0, 0.0, 0.0
        )
        source_axis_x, source_axis_y, source_axis_z = _rotate_vector_by_quaternion_xyzw(
            source_qx, source_qy, source_qz, source_qw, 1.0, 0.0, 0.0
        )
        object_x, object_y, _ = _rotate_vector_by_quaternion_xyzw(
            tf_qx, tf_qy, tf_qz, tf_qw, source_axis_x, source_axis_y, source_axis_z
        )

        if math.hypot(baseline_x, baseline_y) <= 1e-6:
            return 0.0, 0.0, 0.0, 1.0
        if math.hypot(object_x, object_y) <= 1e-6:
            return 0.0, 0.0, 0.0, 1.0

        baseline_yaw = math.atan2(baseline_y, baseline_x)
        object_yaw = math.atan2(object_y, object_x)
        return _quaternion_from_yaw(_wrap_to_pi(object_yaw - baseline_yaw))

    def on_detections(self, msg: Detection3DArray):
        source_frame = str(msg.header.frame_id)
        if not source_frame:
            self.get_logger().warn(
                "Got a 3D detection message with no frame id."
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
            orientation_xyzw = self._transform_in_plane_orientation_delta(
                transform, det.bbox.center.orientation
            )

            if transform is not None:
                x, y, z = self._transform_point(transform, x, y, z)

            if not (self.min_transformed_z <= z <= self.max_transformed_z):
                continue

            det_out = Detection3D()
            det_out.header.stamp = publish_stamp
            det_out.header.frame_id = self.target_frame

            det_out.bbox.center.position.x = x
            det_out.bbox.center.position.y = y
            det_out.bbox.center.position.z = z
            det_out.bbox.center.orientation.x = float(orientation_xyzw[0])
            det_out.bbox.center.orientation.y = float(orientation_xyzw[1])
            det_out.bbox.center.orientation.z = float(orientation_xyzw[2])
            det_out.bbox.center.orientation.w = float(orientation_xyzw[3])
            det_out.bbox.size = det.bbox.size

            for hyp in det.results:
                hyp_out = ObjectHypothesisWithPose()
                hyp_out.hypothesis.class_id = hyp.hypothesis.class_id
                hyp_out.hypothesis.score = hyp.hypothesis.score
                hyp_out.pose.covariance = list(hyp.pose.covariance)
                hyp_out.pose.pose.position.x = x
                hyp_out.pose.pose.position.y = y
                hyp_out.pose.pose.position.z = z
                hyp_out.pose.pose.orientation.x = float(orientation_xyzw[0])
                hyp_out.pose.pose.orientation.y = float(orientation_xyzw[1])
                hyp_out.pose.pose.orientation.z = float(orientation_xyzw[2])
                hyp_out.pose.pose.orientation.w = float(orientation_xyzw[3])
                det_out.results.append(hyp_out)

            out.detections.append(det_out)

        self.pub.publish(out)

        if self.motion_active:
            return

        best = self._select_best_detection(out)
        if best is None:
            return

        best_z = float(best.bbox.center.position.z) + max(
            0.5 * float(best.bbox.size.z), 0.0
        )
        best_position = (
            float(best.bbox.center.position.x),
            float(best.bbox.center.position.y),
            best_z,
        )
        best_position = self._filter_best_position(best_position)
        best_x, best_y, best_z = self._stabilize_best_position(best_position)
        best_orientation = best.bbox.center.orientation
        best_qx = float(best_orientation.x)
        best_qy = float(best_orientation.y)
        best_qz = float(best_orientation.z)
        best_qw = float(best_orientation.w)

        if self.best_pose_pub is not None:
            pose = PoseStamped()
            pose.header.stamp = publish_stamp
            pose.header.frame_id = out.header.frame_id
            pose.pose.position.x = best_x
            pose.pose.position.y = best_y
            pose.pose.position.z = best_z
            pose.pose.orientation.x = best_qx
            pose.pose.orientation.y = best_qy
            pose.pose.orientation.z = best_qz
            pose.pose.orientation.w = best_qw
            self.best_pose_pub.publish(pose)

        if self.best_class_pub is not None and best.results:
            msg_out = String()
            msg_out.data = str(best.results[0].hypothesis.class_id)
            self.best_class_pub.publish(msg_out)

        if self.tf_broadcaster is not None:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = publish_stamp
            tf_msg.header.frame_id = out.header.frame_id
            tf_msg.child_frame_id = self.best_tf_child_frame
            tf_msg.transform.translation.x = best_x
            tf_msg.transform.translation.y = best_y
            tf_msg.transform.translation.z = best_z
            tf_msg.transform.rotation.x = best_qx
            tf_msg.transform.rotation.y = best_qy
            tf_msg.transform.rotation.z = best_qz
            tf_msg.transform.rotation.w = best_qw
            self.tf_broadcaster.sendTransform(tf_msg)

    def _select_best_detection(self, msg: Detection3DArray) -> Optional[Detection3D]:
        best_det = None
        best_distance = float("inf")
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
            center = det.bbox.center.position
            distance = math.sqrt(
                float(center.x) * float(center.x)
                + float(center.y) * float(center.y)
                + float(center.z) * float(center.z)
            )
            if best_det is None or distance < best_distance:
                best_det = det
                best_distance = distance
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
