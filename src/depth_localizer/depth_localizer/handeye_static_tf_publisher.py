#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

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
    TF_LOOKUP_TIMEOUT_SEC = 0.2

    def __init__(self):
        super().__init__("handeye_static_tf_publisher")

        self.declare_parameter(
            "handeye_result_file",
            "calibration/eye_in_hand_charuco/handeye_result.json",
        )
        self.declare_parameter("child_frame", "")

        result_path = Path(
            str(self.get_parameter("handeye_result_file").value)
        ).expanduser()
        self.output_child_frame = str(self.get_parameter("child_frame").value).strip()

        payload = self._load_json(result_path)
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        self.parent_frame = str(metadata.get("tool_frame", "tool0"))
        self.camera_frame = str((metadata.get("camera") or {}).get("image_frame", ""))

        best_result = (
            payload.get("best_result", {}) if isinstance(payload, dict) else {}
        )
        tool_t_camera = best_result.get("tool_T_camera_frame", {})
        translation = tool_t_camera.get("translation_xyz")
        quaternion = tool_t_camera.get("quaternion_xyzw")

        if not self.camera_frame:
            raise ValueError(
                "handeye_result.json is missing metadata.camera.image_frame"
            )
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

        self.output_child_frame = self.output_child_frame or self.camera_frame
        self.tool_t_camera = self._matrix_from_translation_quaternion(
            translation, quaternion
        )

        self._broadcaster = StaticTransformBroadcaster(self)
        self._tf_buffer = None
        self._tf_listener = None
        self._timer = None

        self.get_logger().info(
            f"Loaded hand-eye result {self.parent_frame} -> {self.camera_frame} "
            f"from {result_path}"
        )

        if self.output_child_frame == self.camera_frame:
            self._publish_transform(self.tool_t_camera, "direct")
            return

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._timer = self.create_timer(0.1, self._publish_composed_transform)
        self.get_logger().info(
            f"Waiting for TF {self.camera_frame} <- {self.output_child_frame} "
            "to compose the calibrated hand-eye transform."
        )

    def _publish_composed_transform(self) -> None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self.camera_frame,
                self.output_child_frame,
                Time(),
                timeout=Duration(seconds=self.TF_LOOKUP_TIMEOUT_SEC),
            )
        except TransformException:
            return

        tool_t_child = self.tool_t_camera @ self._matrix_from_transform(transform)
        self._publish_transform(
            tool_t_child,
            f"composed via TF {self.camera_frame} <- {self.output_child_frame}",
        )
        self._timer.cancel()

    def _publish_transform(self, matrix: np.ndarray, detail: str) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.output_child_frame

        transform.transform.translation.x = float(matrix[0, 3])
        transform.transform.translation.y = float(matrix[1, 3])
        transform.transform.translation.z = float(matrix[2, 3])

        qx, qy, qz, qw = self._quaternion_from_matrix(matrix[:3, :3])
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self._broadcaster.sendTransform(transform)
        self.get_logger().info(
            f"Published /tf_static: {self.parent_frame} -> {self.output_child_frame} "
            f"({detail})"
        )

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
        translation = transform_stamped.transform.translation
        rotation = transform_stamped.transform.rotation
        return cls._matrix_from_translation_quaternion(
            (float(translation.x), float(translation.y), float(translation.z)),
            (
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
                float(rotation.w),
            ),
        )

    @classmethod
    def _quaternion_from_matrix(
        cls, rotation: np.ndarray
    ) -> tuple[float, float, float, float]:
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

        return cls._normalize_quaternion_xyzw(
            float(qx), float(qy), float(qz), float(qw)
        )

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
