#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

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

ARUCO_DICTIONARIES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
}

DEFAULT_IMAGE_TOPIC = "/realsense_cam/color/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/realsense_cam/color/camera_info"
DEFAULT_BASE_FRAME = "base"
DEFAULT_TOOL_FRAME = "tool0"
DEFAULT_SESSION_DIR = "calibration/eye_in_hand_charuco"
DEFAULT_BOARD_TYPE = "charuco"
DEFAULT_BOARD_COLS = 7
DEFAULT_BOARD_ROWS = 5
DEFAULT_SQUARE_SIZE_M = 0.0325
DEFAULT_MARKER_SIZE_M = 0.025
DEFAULT_ARUCO_DICT = "5x5_100"
DEFAULT_MIN_CHARUCO_CORNERS = 6


@dataclass
class CollectorConfig:
    image_topic: str
    camera_info_topic: str
    base_frame: str
    tool_frame: str
    board_type: str
    board_cols: int
    board_rows: int
    square_size_m: float
    marker_size_m: float | None
    aruco_dict: str
    min_charuco_corners: int
    session_dir: Path
    preview_scale: float = 1.5
    max_tf_delta_ms: float = 50.0
    allow_latest_tf_fallback: bool = False
    exact_tf_wait_sec: float = 0.3
    tf_poll_interval_sec: float = 0.02
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
        self.latest_overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        self.recent_frames = deque(maxlen=30)
        self.aruco_dictionary = None
        self.charuco_board = None
        self.charuco_detector = None

        self.images_dir = ensure_directory(config.session_dir / "images")
        self.sample_file = config.session_dir / "samples.json"
        self.samples = []
        self.session_metadata = {}
        self._load_existing_session()

        existing_board = self.session_metadata.get("board", {})
        if existing_board:
            mismatches = self._board_metadata_mismatches(existing_board)
            if mismatches:
                raise ValueError(
                    "Existing session metadata does not match the requested target settings. "
                    "Use a fresh --session-dir or align the collector arguments. "
                    f"Mismatches: {', '.join(mismatches)}"
                )

        if config.board_type == "charuco":
            if config.marker_size_m is None:
                raise ValueError("marker_size_m is required for a ChArUco board.")
            if config.marker_size_m >= config.square_size_m:
                raise ValueError("marker_size_m must be smaller than square_size_m.")
            self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(
                ARUCO_DICTIONARIES[config.aruco_dict]
            )
            self.charuco_board = cv2.aruco.CharucoBoard(
                (config.board_cols, config.board_rows),
                config.square_size_m,
                config.marker_size_m,
                self.aruco_dictionary,
            )
            self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board)
        elif config.board_cols % 2 == 0 and config.board_rows % 2 == 0:
            self.get_logger().warn(
                "The checkerboard has even by even inner corners. "
                f"It is {config.board_cols} by {config.board_rows}. "
                "This can make the board pose flip even when the image looks right."
            )

        self.create_subscription(Image, config.image_topic, self.on_image, 10)
        self.create_subscription(
            CameraInfo, config.camera_info_topic, self.on_camera_info, 10
        )

    def _load_existing_session(self):
        if not self.sample_file.exists():
            return

        payload = load_json(self.sample_file)
        self.session_metadata = payload.get("metadata", {})
        self.samples = payload.get("samples", [])
        camera = self.session_metadata.get("camera", {})
        if camera.get("camera_matrix") is not None:
            self.camera_matrix = np.asarray(camera["camera_matrix"], dtype=np.float64)
        if camera.get("dist_coeffs") is not None:
            self.dist_coeffs = np.asarray(camera["dist_coeffs"], dtype=np.float64)
        self.image_frame = str(camera.get("image_frame", ""))

    def _board_metadata_mismatches(self, existing_board: dict) -> list[str]:
        config = self.config
        mismatches = []

        comparisons = [
            ("type", str(existing_board.get("type", "chessboard")), config.board_type),
            ("cols", int(existing_board.get("cols", 0) or 0), config.board_cols),
            ("rows", int(existing_board.get("rows", 0) or 0), config.board_rows),
        ]
        for label, existing, requested in comparisons:
            if existing != requested:
                mismatches.append(
                    f"{label}={existing} (existing) vs {requested} (requested)"
                )

        existing_square_size = float(existing_board.get("square_size_m", 0.0) or 0.0)
        if abs(existing_square_size - config.square_size_m) > 1e-9:
            mismatches.append(
                "square_size_m="
                f"{existing_board.get('square_size_m')} (existing) vs "
                f"{config.square_size_m} (requested)"
            )

        if config.board_type == "charuco":
            existing_marker_size = float(
                existing_board.get("marker_size_m", 0.0) or 0.0
            )
            requested_marker_size = float(config.marker_size_m or 0.0)
            if abs(existing_marker_size - requested_marker_size) > 1e-9:
                mismatches.append(
                    "marker_size_m="
                    f"{existing_board.get('marker_size_m')} (existing) vs "
                    f"{config.marker_size_m} (requested)"
                )
            existing_dict = str(existing_board.get("aruco_dict", ""))
            if existing_dict != config.aruco_dict:
                mismatches.append(
                    f"aruco_dict={existing_dict} (existing) vs "
                    f"{config.aruco_dict} (requested)"
                )

        return mismatches

    @staticmethod
    def _detection_result(
        *,
        found: bool,
        display_text: str,
        pose_object_points=None,
        pose_image_points=None,
        sample_metadata: dict | None = None,
        **extra,
    ) -> dict:
        return {
            "found": found,
            "display_text": display_text,
            "pose_object_points": pose_object_points,
            "pose_image_points": pose_image_points,
            "sample_metadata": sample_metadata or {},
            **extra,
        }

    @staticmethod
    def _history_frame(
        *,
        bgr: np.ndarray,
        image_msg: Image,
        detection: dict,
        overlay: np.ndarray,
    ) -> dict:
        pose_object_points = detection.get("pose_object_points")
        pose_image_points = detection.get("pose_image_points")
        return {
            "bgr": bgr.copy(),
            "image_msg": image_msg,
            "pose_object_points": (
                np.asarray(pose_object_points, dtype=np.float32).copy()
                if pose_object_points is not None
                else None
            ),
            "pose_image_points": (
                np.asarray(pose_image_points, dtype=np.float32).copy()
                if pose_image_points is not None
                else None
            ),
            "sample_metadata": dict(detection["sample_metadata"]),
            "overlay": overlay.copy(),
        }

    def on_camera_info(self, msg: CameraInfo):
        self.camera_matrix = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(msg.d, dtype=np.float64)
        self.image_frame = msg.header.frame_id

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"I could not convert the image. {exc}")
            return

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detection = self.detect_target(gray)
        self.latest_overlay = self.draw_overlay(bgr, detection)
        self.recent_frames.append(
            self._history_frame(
                bgr=bgr,
                image_msg=msg,
                detection=detection,
                overlay=self.latest_overlay,
            )
        )

    def find_chessboard(
        self, gray_image: np.ndarray
    ) -> tuple[bool, np.ndarray | None, tuple[int, int] | None]:
        pattern_size = (self.config.board_cols, self.config.board_rows)

        image_variants = [gray_image]
        image_variants.append(cv2.equalizeHist(gray_image))

        for variant in image_variants:
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
            corners = cv2.cornerSubPix(variant, corners, (11, 11), (-1, -1), criteria)
            return True, corners, pattern_size

        return False, None, None

    def find_charuco(self, gray_image: np.ndarray) -> dict:
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            self.charuco_detector.detectBoard(gray_image)
        )

        if charuco_ids is None or charuco_corners is None:
            return self._detection_result(
                found=False,
                display_text="charuco not found",
                marker_corners=marker_corners,
                marker_ids=marker_ids,
                charuco_corners=None,
                charuco_ids=None,
                sample_metadata={"board_type": "charuco"},
            )

        corner_count = int(len(charuco_ids))
        if corner_count < self.config.min_charuco_corners:
            return self._detection_result(
                found=False,
                display_text=(
                    f"charuco {corner_count} corners < {self.config.min_charuco_corners}"
                ),
                marker_corners=marker_corners,
                marker_ids=marker_ids,
                charuco_corners=charuco_corners,
                charuco_ids=charuco_ids,
                sample_metadata={
                    "board_type": "charuco",
                    "charuco_corner_count": corner_count,
                },
            )

        if self.charuco_board.checkCharucoCornersCollinear(charuco_ids):
            return self._detection_result(
                found=False,
                display_text="charuco corners are collinear",
                marker_corners=marker_corners,
                marker_ids=marker_ids,
                charuco_corners=charuco_corners,
                charuco_ids=charuco_ids,
                sample_metadata={
                    "board_type": "charuco",
                    "charuco_corner_count": corner_count,
                },
            )

        object_points, image_points = self.charuco_board.matchImagePoints(
            charuco_corners, charuco_ids
        )
        return self._detection_result(
            found=True,
            display_text=f"charuco found {corner_count} corners",
            pose_object_points=object_points,
            pose_image_points=image_points,
            marker_corners=marker_corners,
            marker_ids=marker_ids,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
            sample_metadata={
                "board_type": "charuco",
                "charuco_corner_count": corner_count,
                "charuco_ids": np.asarray(charuco_ids, dtype=np.int32)
                .reshape(-1)
                .astype(int)
                .tolist(),
            },
        )

    def detect_target(self, gray_image: np.ndarray) -> dict:
        if self.config.board_type == "charuco":
            return self.find_charuco(gray_image)

        found, corners, pattern_size = self.find_chessboard(gray_image)
        if not found:
            return self._detection_result(
                found=False,
                display_text="board not found",
                corners=None,
                pattern_size=None,
                sample_metadata={"board_type": "chessboard"},
            )

        return self._detection_result(
            found=True,
            display_text=f"board found {pattern_size[0]}x{pattern_size[1]}",
            pose_object_points=build_object_points(
                pattern_size[0], pattern_size[1], self.config.square_size_m
            ),
            pose_image_points=corners,
            corners=corners,
            pattern_size=pattern_size,
            sample_metadata={
                "board_type": "chessboard",
                "pattern_size": list(pattern_size),
            },
        )

    def draw_overlay(
        self,
        bgr_image: np.ndarray,
        detection: dict,
    ) -> np.ndarray:
        overlay = bgr_image.copy()
        found = bool(detection.get("found", False))
        if detection.get("sample_metadata", {}).get("board_type") == "charuco":
            marker_corners = detection.get("marker_corners")
            marker_ids = detection.get("marker_ids")
            charuco_corners = detection.get("charuco_corners")
            charuco_ids = detection.get("charuco_ids")
            if marker_corners is not None and len(marker_corners) > 0:
                cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
            if (
                charuco_corners is not None
                and charuco_ids is not None
                and len(charuco_ids) > 0
            ):
                cv2.aruco.drawDetectedCornersCharuco(
                    overlay, charuco_corners, charuco_ids, (0, 220, 0)
                )
        else:
            corners = detection.get("corners")
            pattern_size = detection.get("pattern_size")
            if found and corners is not None:
                cv2.drawChessboardCorners(
                    overlay,
                    pattern_size,
                    corners,
                    True,
                )

        color = (0, 220, 0) if found else (0, 0, 255)
        text = str(detection.get("display_text", "board not found"))
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

    def is_tf_lookup_acceptable(self, tf_lookup: dict | None) -> tuple[bool, str]:
        if tf_lookup is None:
            return False, "No TF metadata was recorded for this sample."
        if tf_lookup["used_latest_tf"] and not self.config.allow_latest_tf_fallback:
            detail = tf_lookup.get("exact_lookup_error")
            if detail:
                return (
                    False,
                    "TF lookup fell back to latest transform instead of image timestamp. "
                    f"Exact lookup error: {detail}",
                )
            return (
                False,
                "TF lookup fell back to latest transform instead of image timestamp.",
            )
        if abs(tf_lookup["delta_ms"]) > self.config.max_tf_delta_ms:
            return (
                False,
                f"Image/TF timestamp delta {tf_lookup['delta_ms']:.1f} ms exceeds "
                f"{self.config.max_tf_delta_ms:.1f} ms.",
            )
        return True, ""

    @staticmethod
    def stamp_to_nanoseconds(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def lookup_base_t_tool(self, stamp) -> tuple[np.ndarray, dict] | tuple[None, None]:
        exact_lookup_error = ""
        for query_time in [Time.from_msg(stamp), Time()]:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.config.base_frame,
                    self.config.tool_frame,
                    query_time,
                    timeout=Duration(seconds=0.0),
                )
                image_ns = self.stamp_to_nanoseconds(stamp)
                tf_ns = self.stamp_to_nanoseconds(transform.header.stamp)
                return transform_msg_to_matrix(transform.transform), {
                    "requested_image_stamp": {
                        "sec": int(stamp.sec),
                        "nanosec": int(stamp.nanosec),
                    },
                    "used_tf_stamp": {
                        "sec": int(transform.header.stamp.sec),
                        "nanosec": int(transform.header.stamp.nanosec),
                    },
                    "delta_ms": float((tf_ns - image_ns) / 1_000_000.0),
                    "used_latest_tf": bool(query_time.nanoseconds == 0),
                    "exact_lookup_error": exact_lookup_error,
                }
            except TransformException as exc:
                if query_time.nanoseconds != 0 and not exact_lookup_error:
                    exact_lookup_error = str(exc)
                continue
        return None, None

    def select_frame_for_save(self):
        recent_frames = list(self.recent_frames)
        if not recent_frames:
            return None, None, None, "No image received yet."

        deadline = time.monotonic() + self.config.exact_tf_wait_sec
        last_rejection = ""
        while True:
            for age, frame in enumerate(reversed(recent_frames)):
                if (
                    frame["pose_object_points"] is None
                    or frame["pose_image_points"] is None
                ):
                    continue

                base_t_tool, tf_lookup = self.lookup_base_t_tool(
                    frame["image_msg"].header.stamp
                )
                if base_t_tool is None:
                    last_rejection = (
                        f"Could not resolve TF {self.config.base_frame} <- "
                        f"{self.config.tool_frame}."
                    )
                    continue

                tf_ok, tf_error = self.is_tf_lookup_acceptable(tf_lookup)
                if not tf_ok:
                    last_rejection = tf_error
                    continue

                if age > 0:
                    self.get_logger().info(
                        f"Using a recent frame {age} images older than the live preview "
                        "to match TF at the exact image timestamp."
                    )
                return frame, base_t_tool, tf_lookup, ""

            if time.monotonic() >= deadline:
                return None, None, None, last_rejection

            rclpy.spin_once(self, timeout_sec=self.config.tf_poll_interval_sec)

    def save_current_sample(self):
        if not self.recent_frames:
            self.get_logger().warn("No image has arrived yet.")
            return
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn("No camera info has arrived yet.")
            return
        selected_frame, selected_base_t_tool, selected_tf_lookup, last_rejection = (
            self.select_frame_for_save()
        )

        if selected_frame is None:
            if last_rejection:
                self.get_logger().warn(f"Sample was skipped. {last_rejection}")
            else:
                self.get_logger().warn(
                    "Checkerboard not found in any recent frame. Hold the board steady "
                    "and try again."
                )
            return

        solved, rvec, tvec = cv2.solvePnP(
            selected_frame["pose_object_points"],
            selected_frame["pose_image_points"],
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not solved:
            self.get_logger().warn("Could not solve the pose for this sample.")
            return

        sample_index = (
            max((int(sample.get("index", -1)) for sample in self.samples), default=-1)
            + 1
        )
        image_path = self.images_dir / f"sample_{sample_index:04d}.png"
        overlay_image_path = self.images_dir / f"sample_{sample_index:04d}_overlay.png"
        cv2.imwrite(str(image_path), selected_frame["bgr"])
        cv2.imwrite(str(overlay_image_path), selected_frame["overlay"])

        sample = {
            "index": sample_index,
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "image_path": str(image_path.relative_to(self.config.session_dir)),
            "overlay_image_path": str(
                overlay_image_path.relative_to(self.config.session_dir)
            ),
            "image_stamp": {
                "sec": int(selected_frame["image_msg"].header.stamp.sec),
                "nanosec": int(selected_frame["image_msg"].header.stamp.nanosec),
            },
            "tf_lookup": selected_tf_lookup,
            "base_T_tool": pose_dict_from_matrix(selected_base_t_tool),
            "target_to_camera": {
                **selected_frame["sample_metadata"],
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
                        "type": self.config.board_type,
                        "cols": self.config.board_cols,
                        "rows": self.config.board_rows,
                        "square_size_m": self.config.square_size_m,
                        **(
                            {
                                "marker_size_m": self.config.marker_size_m,
                                "aruco_dict": self.config.aruco_dict,
                                "min_charuco_corners": self.config.min_charuco_corners,
                            }
                            if self.config.board_type == "charuco"
                            else {}
                        ),
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

    def print_tf_lookup_summary(self):
        rows = []
        for sample in self.samples:
            tf_lookup = sample.get("tf_lookup")
            if not tf_lookup:
                continue
            rows.append(
                {
                    "index": sample.get("index"),
                    "delta_ms": float(tf_lookup.get("delta_ms", 0.0)),
                    "used_latest_tf": bool(tf_lookup.get("used_latest_tf", False)),
                    "image_path": sample.get("image_path", ""),
                }
            )

        if not rows:
            return

        rows.sort(key=lambda item: abs(item["delta_ms"]), reverse=True)
        deltas = [abs(item["delta_ms"]) for item in rows]

        print()
        print("TF timing summary.")
        print(f"Samples with TF lookup. {len(rows)}")
        print(f"Max absolute delta in ms. {max(deltas):.3f}")
        print(f"Mean absolute delta in ms. {sum(deltas) / len(deltas):.3f}")
        print("Worst samples.")
        for row in rows[:5]:
            print(
                f"Sample {row['index']}. "
                f"Delta ms {row['delta_ms']:.3f}. "
                f"Used latest TF {row['used_latest_tf']}. "
                f"Path {row['image_path']}."
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect eye-in-hand samples. Defaults are tuned for the current "
            "UR3e + RealSense 7x5 ChArUco board workflow."
        )
    )
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--camera-info-topic", default=DEFAULT_CAMERA_INFO_TOPIC)
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    parser.add_argument("--tool-frame", default=DEFAULT_TOOL_FRAME)
    parser.add_argument(
        "--board-type",
        choices=["chessboard", "charuco"],
        default=DEFAULT_BOARD_TYPE,
        help="Calibration target type. Defaults to your current ChArUco setup.",
    )
    parser.add_argument("--board-cols", type=int, default=DEFAULT_BOARD_COLS)
    parser.add_argument("--board-rows", type=int, default=DEFAULT_BOARD_ROWS)
    parser.add_argument("--square-size-m", type=float, default=DEFAULT_SQUARE_SIZE_M)
    parser.add_argument(
        "--marker-size-m",
        type=float,
        default=DEFAULT_MARKER_SIZE_M,
        help="Marker side length for ChArUco boards.",
    )
    parser.add_argument(
        "--aruco-dict",
        choices=sorted(ARUCO_DICTIONARIES.keys()),
        default=DEFAULT_ARUCO_DICT,
        help="ArUco dictionary for ChArUco boards.",
    )
    parser.add_argument(
        "--min-charuco-corners",
        type=int,
        default=DEFAULT_MIN_CHARUCO_CORNERS,
        help="Minimum detected ChArUco corners required to save a sample.",
    )
    parser.add_argument(
        "--session-dir",
        default=DEFAULT_SESSION_DIR,
        help="Session folder. Defaults to the current ChArUco calibration session path.",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.5,
        help="Scale factor for the preview window image, e.g. 2.0 makes it twice as large.",
    )
    parser.add_argument(
        "--max-tf-delta-ms",
        type=float,
        default=50.0,
        help="Maximum allowed absolute time difference between image and TF pose.",
    )
    parser.add_argument(
        "--allow-latest-tf-fallback",
        action="store_true",
        help="Allow saving samples even when TF at the exact image timestamp is unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = CollectorConfig(
        image_topic=args.image_topic,
        camera_info_topic=args.camera_info_topic,
        base_frame=args.base_frame,
        tool_frame=args.tool_frame,
        board_type=args.board_type,
        board_cols=args.board_cols,
        board_rows=args.board_rows,
        square_size_m=args.square_size_m,
        marker_size_m=args.marker_size_m if args.board_type == "charuco" else None,
        aruco_dict=args.aruco_dict,
        min_charuco_corners=max(4, args.min_charuco_corners),
        session_dir=Path(args.session_dir),
        preview_scale=max(0.1, args.preview_scale),
        max_tf_delta_ms=max(0.0, args.max_tf_delta_ms),
        allow_latest_tf_fallback=bool(args.allow_latest_tf_fallback),
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
        print("Move the robot to a stable pose, then press s to save a sample.")
        print("Press q to finish collecting.")
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
        node.print_tf_lookup_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
