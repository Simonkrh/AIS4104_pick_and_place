#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import (
    Buffer,
    StaticTransformBroadcaster,
    TransformException,
    TransformListener,
)


class HandeyeStaticTFPublisher(Node):
    def __init__(self):
        super().__init__("handeye_static_tf_publisher")

        self.declare_parameter(
            "handeye_result_file",
            "calibration/eye_in_hand_charuco/handeye_result.json",
        )
        self.declare_parameter("parent_frame", "")
        self.declare_parameter("child_frame", "")
        self.declare_parameter("handeye_camera_frame", "")
        self.declare_parameter("auto_child_frame", True)
        self.declare_parameter("tf_lookup_timeout_sec", 0.2)
        self.declare_parameter("tf_wait_sec", 5.0)

        result_path = Path(
            str(self.get_parameter("handeye_result_file").value)
        ).expanduser()
        parent_override = str(self.get_parameter("parent_frame").value).strip()
        child_override = str(self.get_parameter("child_frame").value).strip()
        camera_frame_override = str(self.get_parameter("handeye_camera_frame").value).strip()
        auto_child_frame = bool(self.get_parameter("auto_child_frame").value)
        self._tf_lookup_timeout_sec = float(
            self.get_parameter("tf_lookup_timeout_sec").value
        )
        self._tf_wait_sec = float(self.get_parameter("tf_wait_sec").value)

        payload = self._load_json(result_path)

        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        default_parent = str(metadata.get("tool_frame", "tool0"))
        handeye_camera_frame = camera_frame_override or str(
            (metadata.get("camera", {}) or {}).get("image_frame", "")
        )

        best_result = payload.get("best_result", {}) if isinstance(payload, dict) else {}
        tool_t_camera = best_result.get("tool_T_camera_frame", {})
        translation = tool_t_camera.get("translation_xyz")
        quaternion = tool_t_camera.get("quaternion_xyzw")

        if translation is None or quaternion is None:
            raise ValueError(
                "handeye_result.json is missing best_result.tool_T_camera_frame "
                "translation_xyz/quaternion_xyzw"
            )
        if len(translation) != 3 or len(quaternion) != 4:
            raise ValueError(
                "handeye_result.json has invalid tool_T_camera_frame sizes: "
                f"translation_xyz={translation}, quaternion_xyzw={quaternion}"
            )

        if parent_override and parent_override != default_parent:
            raise ValueError(
                "parent_frame override is not supported for hand-eye results. "
                "The result file encodes tool_T_camera, so the parent must match "
                f"metadata.tool_frame='{default_parent}'. Got parent_frame='{parent_override}'."
            )

        if not handeye_camera_frame:
            raise ValueError(
                "handeye_camera_frame is empty and metadata.camera.image_frame was not found in the "
                "hand-eye result file."
            )

        self._parent_frame = default_parent
        self._handeye_camera_frame = handeye_camera_frame
        self._tool_t_handeye_camera = self._matrix_from_translation_quaternion(
            translation, quaternion
        )

        self._requested_child_frame = child_override
        self._auto_child_frame = auto_child_frame
        self._auto_child_candidate = (
            self._suggest_camera_link_frame(handeye_camera_frame)
            if (auto_child_frame and not child_override)
            else ""
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._broadcaster = StaticTransformBroadcaster(self)
        self._published = False
        self._start_monotonic = time.monotonic()

        # Publish immediately when possible, otherwise wait for the camera TF tree.
        self._timer = self.create_timer(0.1, self._try_publish)

        requested_child = child_override or (self._auto_child_candidate or handeye_camera_frame)
        self.get_logger().info(
            "Loaded hand-eye result: "
            f"{default_parent} -> {handeye_camera_frame}. "
            f"Will publish /tf_static: {default_parent} -> {requested_child} "
            f"from {result_path}"
        )

    def _try_publish(self):
        if self._published:
            return

        child_frame = self._resolve_output_child_frame()
        if not child_frame:
            return

        tool_t_child = self._tool_t_handeye_camera
        composed = False
        if child_frame != self._handeye_camera_frame:
            try:
                # We have tool_T_handeye_camera from calibration.
                # To publish tool_T_child, we need handeye_camera_T_child.
                # lookup_transform(target, source) returns target_T_source.
                transform = self._tf_buffer.lookup_transform(
                    self._handeye_camera_frame,
                    child_frame,
                    Time(),
                    timeout=Duration(seconds=max(self._tf_lookup_timeout_sec, 0.0)),
                )
            except TransformException:
                return

            handeye_camera_t_child = self._matrix_from_transform(transform)
            tool_t_child = tool_t_child @ handeye_camera_t_child
            composed = True

        transform_msg = TransformStamped()
        transform_msg.header.stamp = self.get_clock().now().to_msg()
        transform_msg.header.frame_id = self._parent_frame
        transform_msg.child_frame_id = child_frame

        transform_msg.transform.translation.x = float(tool_t_child[0, 3])
        transform_msg.transform.translation.y = float(tool_t_child[1, 3])
        transform_msg.transform.translation.z = float(tool_t_child[2, 3])

        qx, qy, qz, qw = self._quaternion_from_matrix(tool_t_child[:3, :3])
        transform_msg.transform.rotation.x = float(qx)
        transform_msg.transform.rotation.y = float(qy)
        transform_msg.transform.rotation.z = float(qz)
        transform_msg.transform.rotation.w = float(qw)

        self._broadcaster.sendTransform(transform_msg)
        self._published = True
        self._timer.cancel()

        detail = "direct"
        if composed:
            detail = f"composed via TF {self._handeye_camera_frame} <- {child_frame}"
        self.get_logger().info(
            "Published /tf_static: "
            f"{self._parent_frame} -> {child_frame} ({detail})"
        )

    def _resolve_output_child_frame(self) -> str:
        if self._requested_child_frame:
            return self._requested_child_frame

        if not self._auto_child_candidate:
            return self._handeye_camera_frame

        # If the candidate exists in TF, prefer it. Otherwise wait a bit and fall back to
        # the calibration camera frame (which is safe when the camera doesn't publish TF).
        try:
            self._tf_buffer.lookup_transform(
                self._handeye_camera_frame,
                self._auto_child_candidate,
                Time(),
                timeout=Duration(seconds=max(self._tf_lookup_timeout_sec, 0.0)),
            )
            return self._auto_child_candidate
        except TransformException:
            pass

        if time.monotonic() - self._start_monotonic >= max(self._tf_wait_sec, 0.0):
            self.get_logger().warn(
                "Could not resolve TF between the calibration camera frame and the auto-selected "
                f"child frame '{self._auto_child_candidate}'. Falling back to publishing "
                f"{self._parent_frame} -> {self._handeye_camera_frame} directly. "
                "If you want to keep the RealSense TF tree enabled, consider setting "
                "child_frame:=realsense_cam_link (or disabling camera publish_tf)."
            )
            self._auto_child_candidate = ""
            return self._handeye_camera_frame

        return ""

    @staticmethod
    def _suggest_camera_link_frame(handeye_camera_frame: str) -> str:
        suffixes = (
            "_color_optical_frame",
            "_depth_optical_frame",
            "_infra_optical_frame",
            "_confidence_optical_frame",
        )
        for suffix in suffixes:
            if handeye_camera_frame.endswith(suffix):
                prefix = handeye_camera_frame[: -len(suffix)]
                if prefix:
                    return f"{prefix}_link"
        return ""

    @staticmethod
    def _normalize_quaternion_xyzw(
        x: float, y: float, z: float, w: float
    ) -> tuple[float, float, float, float]:
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm <= 0.0:
            return 0.0, 0.0, 0.0, 1.0
        return x / norm, y / norm, z / norm, w / norm

    @classmethod
    def _rotation_matrix_from_quaternion(
        cls, qx: float, qy: float, qz: float, qw: float
    ) -> np.ndarray:
        qx, qy, qz, qw = cls._normalize_quaternion_xyzw(qx, qy, qz, qw)
        xx = qx * qx
        yy = qy * qy
        zz = qz * qz
        xy = qx * qy
        xz = qx * qz
        yz = qy * qz
        wx = qw * qx
        wy = qw * qy
        wz = qw * qz
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    @classmethod
    def _matrix_from_translation_quaternion(
        cls, translation_xyz, quaternion_xyzw
    ) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = cls._rotation_matrix_from_quaternion(
            float(quaternion_xyzw[0]),
            float(quaternion_xyzw[1]),
            float(quaternion_xyzw[2]),
            float(quaternion_xyzw[3]),
        )
        matrix[:3, 3] = np.asarray(translation_xyz, dtype=np.float64).reshape(3)
        return matrix

    @classmethod
    def _matrix_from_transform(cls, transform_stamped) -> np.ndarray:
        t = transform_stamped.transform.translation
        q = transform_stamped.transform.rotation
        return cls._matrix_from_translation_quaternion(
            (float(t.x), float(t.y), float(t.z)),
            (float(q.x), float(q.y), float(q.z), float(q.w)),
        )

    @classmethod
    def _quaternion_from_matrix(cls, rotation: np.ndarray) -> tuple[float, float, float, float]:
        m = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        trace = float(m[0, 0] + m[1, 1] + m[2, 2])

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + float(m[0, 0]) - float(m[1, 1]) - float(m[2, 2])) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + float(m[1, 1]) - float(m[0, 0]) - float(m[2, 2])) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + float(m[2, 2]) - float(m[0, 0]) - float(m[1, 1])) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

        qx, qy, qz, qw = cls._normalize_quaternion_xyzw(float(qx), float(qy), float(qz), float(qw))
        return qx, qy, qz, qw

    @staticmethod
    def _load_json(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Hand-eye result file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def main():
    rclpy.init()
    node = HandeyeStaticTFPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
