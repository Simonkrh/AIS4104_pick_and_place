#!/usr/bin/env python3
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray


def _make_qos(
    reliability: ReliabilityPolicy,
    durability: DurabilityPolicy,
    depth: int,
) -> QoSProfile:
    return QoSProfile(
        reliability=reliability,
        durability=durability,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class Detection3DNode(Node):
    def __init__(self):
        super().__init__("detection_3d_node")

        self.declare_parameter("detection_topic", "/yolo/detections")
        self.declare_parameter(
            "depth_topic", "/realsense_cam/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("camera_info_topic", "/realsense_cam/color/camera_info")
        self.declare_parameter("output_topic", "/yolo/detections_3d")
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("roi_half_size", 2)
        self.declare_parameter("min_depth_m", 0.10)
        self.declare_parameter("max_depth_m", 2.00)
        self.declare_parameter("enable_temporal_filter", True)
        self.declare_parameter("smoothing_alpha", 0.4)
        self.declare_parameter("max_jump_m", 0.08)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.10)
        self.declare_parameter("max_depth_age_sec", 0.75)

        detection_topic = str(self.get_parameter("detection_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.roi_half_size = int(self.get_parameter("roi_half_size").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.enable_temporal_filter = bool(
            self.get_parameter("enable_temporal_filter").value
        )
        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        self.max_jump_m = float(self.get_parameter("max_jump_m").value)
        self.max_depth_age_sec = float(self.get_parameter("max_depth_age_sec").value)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Detection3DArray, output_topic, 10)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_width = None
        self.camera_height = None
        self.camera_frame_id = ""
        self.track_state = {}
        self.latest_depth_msg = None
        self.latest_depth_received_ns = None
        self._last_depth_warn_ns = 0
        self._logged_first_depth = False
        self._logged_first_camera_info = False

        # Camera drivers typically publish image and camera_info with sensor-data QoS
        # (best effort, volatile). Matching that profile avoids silently dropping frames.
        camera_info_qos = qos_profile_sensor_data
        depth_qos = qos_profile_sensor_data
        detection_qos = _make_qos(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.create_subscription(
            CameraInfo, camera_info_topic, self.on_camera_info, camera_info_qos
        )
        self.create_subscription(Image, depth_topic, self.on_depth, depth_qos)
        self.create_subscription(
            Detection2DArray, detection_topic, self.on_detection, detection_qos
        )

        self.get_logger().info(f"Subscribing detections: {detection_topic}")
        self.get_logger().info(f"Subscribing depth: {depth_topic}")
        self.get_logger().info(f"Subscribing camera info: {camera_info_topic}")
        self.get_logger().info(f"Publishing 3D detections: {output_topic}")

    def on_camera_info(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self.camera_width = int(msg.width)
        self.camera_height = int(msg.height)
        self.camera_frame_id = msg.header.frame_id
        if not self._logged_first_camera_info:
            self._logged_first_camera_info = True
            self.get_logger().info(
                "Received first camera info on "
                f"{self.camera_info_topic} ({msg.width}x{msg.height}, frame_id={msg.header.frame_id or '<empty>'})"
            )

    def on_depth(self, msg: Image):
        self.latest_depth_msg = msg
        self.latest_depth_received_ns = self.get_clock().now().nanoseconds
        if not self._logged_first_depth:
            self._logged_first_depth = True
            self.get_logger().info(
                "Received first depth frame on "
                f"{self.depth_topic} ({msg.encoding}, {msg.width}x{msg.height}, frame_id={msg.header.frame_id or '<empty>'})"
            )

    def on_detection(self, det_msg: Detection2DArray):
        if self.fx is None or self.fy is None:
            return

        depth_msg = self.latest_depth_msg
        if depth_msg is None or not self._depth_is_fresh():
            self._warn_depth_unavailable()
            return

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        height, width = depth_image.shape[:2]

        scale_u = 1.0
        scale_v = 1.0
        if self.camera_width and self.camera_width > 0:
            scale_u = width / float(self.camera_width)
        if self.camera_height and self.camera_height > 0:
            scale_v = height / float(self.camera_height)

        out = Detection3DArray()
        out.header = det_msg.header
        out.header.frame_id = depth_msg.header.frame_id or self.camera_frame_id

        converted = 0
        for det in det_msg.detections:
            # Map 2D detections into the depth image resolution when color and
            # aligned depth are published at different sizes.
            u = int(round(det.bbox.center.position.x * scale_u))
            v = int(round(det.bbox.center.position.y * scale_v))

            u0 = max(0, u - self.roi_half_size)
            u1 = min(width, u + self.roi_half_size + 1)
            v0 = max(0, v - self.roi_half_size)
            v1 = min(height, v + self.roi_half_size + 1)
            if u0 >= u1 or v0 >= v1:
                continue

            patch = depth_image[v0:v1, u0:u1]
            depth_values_m = self.depth_patch_to_meters(patch, depth_msg.encoding)
            if depth_values_m.size == 0:
                continue

            z = float(np.median(depth_values_m))
            if not (self.min_depth_m <= z <= self.max_depth_m):
                continue

            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy
            label = (
                det.results[0].hypothesis.class_id
                if det.results
                else f"unknown_{converted}"
            )
            x, y, z = self.apply_temporal_filter(label, x, y, z)

            det3d = Detection3D()
            det3d.header = out.header
            det3d.results = det.results
            det3d.bbox.center.position.x = x
            det3d.bbox.center.position.y = y
            det3d.bbox.center.position.z = z
            det3d.bbox.center.orientation.w = 1.0
            det3d.bbox.size.x = max((det.bbox.size_x * z) / self.fx, 0.0)
            det3d.bbox.size.y = max((det.bbox.size_y * z) / self.fy, 0.0)
            det3d.bbox.size.z = 0.05

            for hyp in det3d.results:
                hyp.pose.pose.position.x = x
                hyp.pose.pose.position.y = y
                hyp.pose.pose.position.z = z
                hyp.pose.pose.orientation.w = 1.0

            out.detections.append(det3d)
            converted += 1

        self.pub.publish(out)
        if converted > 0:
            self.get_logger().debug(f"Published {converted} detections with 3D points")

    def _depth_is_fresh(self) -> bool:
        if self.latest_depth_received_ns is None:
            return False
        age_sec = (
            self.get_clock().now().nanoseconds - self.latest_depth_received_ns
        ) / 1_000_000_000.0
        return age_sec <= self.max_depth_age_sec

    def _warn_depth_unavailable(self):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_depth_warn_ns < 1_000_000_000:
            return
        self._last_depth_warn_ns = now_ns
        if self.latest_depth_msg is None:
            self.get_logger().warn(
                f"No depth frame received yet on {self.depth_topic}; skipping 3D localization."
            )
            return
        age_sec = (
            now_ns - self.latest_depth_received_ns
        ) / 1_000_000_000.0
        self.get_logger().warn(
            f"Latest depth frame is stale ({age_sec:.2f}s old); skipping 3D localization."
        )

    def apply_temporal_filter(
        self, label: str, x: float, y: float, z: float
    ) -> tuple[float, float, float]:
        if not self.enable_temporal_filter:
            return x, y, z

        prev = self.track_state.get(label)
        if prev is None:
            self.track_state[label] = (x, y, z)
            return x, y, z

        dx = x - prev[0]
        dy = y - prev[1]
        dz = z - prev[2]
        if np.sqrt(dx * dx + dy * dy + dz * dz) > self.max_jump_m:
            return prev

        alpha = min(max(self.smoothing_alpha, 0.0), 1.0)
        xf = alpha * x + (1.0 - alpha) * prev[0]
        yf = alpha * y + (1.0 - alpha) * prev[1]
        zf = alpha * z + (1.0 - alpha) * prev[2]
        self.track_state[label] = (xf, yf, zf)
        return xf, yf, zf

    def depth_patch_to_meters(self, patch: np.ndarray, encoding: str) -> np.ndarray:
        if patch.size == 0:
            return np.array([], dtype=np.float32)

        if encoding in ("16UC1", "mono16"):
            meters = patch.astype(np.float32) * self.depth_scale
        else:
            meters = patch.astype(np.float32)

        valid = np.isfinite(meters) & (meters > 0.0)
        if not np.any(valid):
            return np.array([], dtype=np.float32)
        return meters[valid]


def main():
    rclpy.init()
    node = Detection3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
