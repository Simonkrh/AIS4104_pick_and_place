#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from handeye_utils import (
    build_object_points,
    ensure_directory,
    load_json,
    pose_dict_from_matrix,
    save_json,
    transform_msg_to_matrix,
)


@dataclass
class CollectorConfig:
    image_topic: str
    camera_info_topic: str
    base_frame: str
    tool_frame: str
    board_cols: int
    board_rows: int
    square_size_m: float
    session_dir: Path
    preview_scale: float = 1.5
    preview_window: str = "Eye-In-Hand Calibration"


class SampleCollector(Node):
    def __init__(self, config: CollectorConfig):
        super().__init__("eye_in_hand_sample_collector")
        self.config = config
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_frame = ""
        self.latest_bgr = None
        self.latest_image_msg = None
        self.latest_corners = None
        self.latest_pattern_size = None
        self.latest_overlay = np.zeros((480, 640, 3), dtype=np.uint8)

        self.images_dir = ensure_directory(config.session_dir / "images")
        self.sample_file = config.session_dir / "samples.json"
        self.samples = []
        self._load_existing_session()

        self.create_subscription(Image, config.image_topic, self.on_image, 10)
        self.create_subscription(
            CameraInfo, config.camera_info_topic, self.on_camera_info, 10
        )

    def _load_existing_session(self):
        if not self.sample_file.exists():
            return

        payload = load_json(self.sample_file)
        self.samples = payload.get("samples", [])
        camera = payload.get("metadata", {}).get("camera", {})
        if camera.get("camera_matrix") is not None:
            self.camera_matrix = np.asarray(camera["camera_matrix"], dtype=np.float64)
        if camera.get("dist_coeffs") is not None:
            self.dist_coeffs = np.asarray(camera["dist_coeffs"], dtype=np.float64)
        self.image_frame = str(camera.get("image_frame", ""))

    def on_camera_info(self, msg: CameraInfo):
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)
        self.image_frame = msg.header.frame_id

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        self.latest_bgr = bgr
        self.latest_image_msg = msg
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        found, corners, pattern_size = self.find_board(gray)
        self.latest_corners = corners if found else None
        self.latest_pattern_size = pattern_size if found else None
        self.latest_overlay = self.draw_overlay(bgr, found, corners, pattern_size)

    def find_board(
        self, gray_image: np.ndarray
    ) -> tuple[bool, np.ndarray | None, tuple[int, int] | None]:
        pattern_sizes = [(self.config.board_cols, self.config.board_rows)]
        swapped = (self.config.board_rows, self.config.board_cols)
        if swapped != pattern_sizes[0]:
            pattern_sizes.append(swapped)

        image_variants = [gray_image]
        image_variants.append(cv2.equalizeHist(gray_image))

        for variant in image_variants:
            for pattern_size in pattern_sizes:
                if hasattr(cv2, "findChessboardCornersSB"):
                    sb_flags = (
                        cv2.CALIB_CB_NORMALIZE_IMAGE
                        | cv2.CALIB_CB_EXHAUSTIVE
                        | cv2.CALIB_CB_ACCURACY
                    )
                    found, corners = cv2.findChessboardCornersSB(
                        variant, pattern_size, flags=sb_flags
                    )
                    if found:
                        return True, corners, pattern_size

                flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                found, corners = cv2.findChessboardCorners(variant, pattern_size, flags)
                if not found:
                    continue

                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners = cv2.cornerSubPix(
                    variant, corners, (11, 11), (-1, -1), criteria
                )
                return True, corners, pattern_size

        return False, None, None

    def draw_overlay(
        self,
        bgr_image: np.ndarray,
        found: bool,
        corners: np.ndarray | None,
        pattern_size: tuple[int, int] | None,
    ) -> np.ndarray:
        overlay = bgr_image.copy()
        if found and corners is not None:
            cv2.drawChessboardCorners(
                overlay,
                pattern_size,
                corners,
                True,
            )

        color = (0, 220, 0) if found else (0, 0, 255)
        if found and pattern_size is not None:
            text = f"board found {pattern_size[0]}x{pattern_size[1]}"
        else:
            text = "board not found"
        cv2.putText(
            overlay,
            f"{text} | samples: {len(self.samples)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        cv2.putText(
            overlay,
            "Press s to save, q to quit.",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return overlay

    def get_preview_image(self) -> np.ndarray:
        if self.config.preview_scale <= 1.0:
            return self.latest_overlay

        width = max(1, int(self.latest_overlay.shape[1] * self.config.preview_scale))
        height = max(1, int(self.latest_overlay.shape[0] * self.config.preview_scale))
        return cv2.resize(
            self.latest_overlay, (width, height), interpolation=cv2.INTER_LINEAR
        )

    def lookup_base_t_tool(self, stamp) -> np.ndarray | None:
        for query_time in [Time.from_msg(stamp), Time()]:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.config.base_frame,
                    self.config.tool_frame,
                    query_time,
                    timeout=Duration(seconds=0.5),
                )
                return transform_msg_to_matrix(transform.transform)
            except TransformException:
                continue
        return None

    def save_current_sample(self):
        if self.latest_bgr is None or self.latest_image_msg is None:
            self.get_logger().warn("No image received yet.")
            return
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn("No camera_info received yet.")
            return
        if self.latest_corners is None:
            self.get_logger().warn("Checkerboard not found in the current image.")
            return
        if self.latest_pattern_size is None:
            self.get_logger().warn(
                "No checkerboard pattern size was stored for this image."
            )
            return

        object_points = build_object_points(
            self.latest_pattern_size[0],
            self.latest_pattern_size[1],
            self.config.square_size_m,
        )

        solved, rvec, tvec = cv2.solvePnP(
            object_points,
            self.latest_corners,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            self.get_logger().warn("solvePnP failed for this sample.")
            return

        base_t_tool = self.lookup_base_t_tool(self.latest_image_msg.header.stamp)
        if base_t_tool is None:
            self.get_logger().warn(
                f"Could not resolve TF {self.config.base_frame} <- {self.config.tool_frame}."
            )
            return

        sample_index = (
            max((int(sample.get("index", -1)) for sample in self.samples), default=-1) + 1
        )
        image_path = self.images_dir / f"sample_{sample_index:04d}.png"
        overlay_image_path = self.images_dir / f"sample_{sample_index:04d}_overlay.png"
        cv2.imwrite(str(image_path), self.latest_bgr)
        cv2.imwrite(str(overlay_image_path), self.latest_overlay)

        sample = {
            "index": sample_index,
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "image_path": str(image_path.relative_to(self.config.session_dir)),
            "overlay_image_path": str(
                overlay_image_path.relative_to(self.config.session_dir)
            ),
            "image_stamp": {
                "sec": int(self.latest_image_msg.header.stamp.sec),
                "nanosec": int(self.latest_image_msg.header.stamp.nanosec),
            },
            "base_T_tool": pose_dict_from_matrix(base_t_tool),
            "target_to_camera": {
                "pattern_size": list(self.latest_pattern_size),
                "rvec": np.asarray(rvec, dtype=np.float64)
                .reshape(3)
                .astype(float)
                .tolist(),
                "tvec_m": np.asarray(tvec, dtype=np.float64)
                .reshape(3)
                .astype(float)
                .tolist(),
            },
        }
        self.samples.append(sample)
        self.write_session()
        self.get_logger().info(f"Saved sample {sample_index}")

    def write_session(self):
        save_json(
            self.sample_file,
            {
                "metadata": {
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "base_frame": self.config.base_frame,
                    "tool_frame": self.config.tool_frame,
                    "image_topic": self.config.image_topic,
                    "camera_info_topic": self.config.camera_info_topic,
                    "board": {
                        "type": "chessboard",
                        "cols": self.config.board_cols,
                        "rows": self.config.board_rows,
                        "square_size_m": self.config.square_size_m,
                    },
                    "camera": {
                        "image_frame": self.image_frame,
                        "camera_matrix": self.camera_matrix.astype(float).tolist()
                        if self.camera_matrix is not None
                        else None,
                        "dist_coeffs": self.dist_coeffs.astype(float)
                        .reshape(-1)
                        .tolist()
                        if self.dist_coeffs is not None
                        else None,
                    },
                },
                "samples": self.samples,
            },
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect eye-in-hand checkerboard samples."
    )
    parser.add_argument("--image-topic", default="/realsense_cam/color/image_raw")
    parser.add_argument(
        "--camera-info-topic", default="/realsense_cam/color/camera_info"
    )
    parser.add_argument("--base-frame", default="base")
    parser.add_argument("--tool-frame", default="tool0")
    parser.add_argument("--board-cols", type=int, default=6)
    parser.add_argument("--board-rows", type=int, default=8)
    parser.add_argument("--square-size-m", type=float, default=0.024)
    parser.add_argument("--session-dir", default="calibration/eye_in_hand")
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.5,
        help="Scale factor for the preview window image, e.g. 2.0 makes it twice as large.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = CollectorConfig(
        image_topic=args.image_topic,
        camera_info_topic=args.camera_info_topic,
        base_frame=args.base_frame,
        tool_frame=args.tool_frame,
        board_cols=args.board_cols,
        board_rows=args.board_rows,
        square_size_m=args.square_size_m,
        session_dir=Path(args.session_dir),
        preview_scale=max(0.1, args.preview_scale),
    )

    ensure_directory(config.session_dir)
    rclpy.init(args=None)
    node = SampleCollector(config)
    cv2.namedWindow(config.preview_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        config.preview_window,
        int(640 * config.preview_scale),
        int(480 * config.preview_scale),
    )

    try:
        print("Move the robot to a stable pose, then press 's' to save a sample.")
        print("Press 'q' to finish collecting.")
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            cv2.imshow(config.preview_window, node.get_preview_image())
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                node.save_current_sample()
            elif key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
