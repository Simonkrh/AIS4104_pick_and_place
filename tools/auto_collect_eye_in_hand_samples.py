#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import time

import rclpy
from moveit_msgs.srv import GetMotionPlan
from pymoveit2 import MoveIt2
from pymoveit2.moveit2 import MoveIt2State
from pymoveit2.robots import ur
from rclpy.callback_groups import ReentrantCallbackGroup

from collect_eye_in_hand_samples import (
    ARUCO_DICTIONARIES,
    DEFAULT_ARUCO_DICT,
    DEFAULT_BASE_FRAME,
    DEFAULT_BOARD_COLS,
    DEFAULT_BOARD_ROWS,
    DEFAULT_BOARD_TYPE,
    DEFAULT_CAMERA_INFO_TOPIC,
    DEFAULT_IMAGE_TOPIC,
    DEFAULT_MARKER_SIZE_M,
    DEFAULT_MIN_CHARUCO_CORNERS,
    DEFAULT_SQUARE_SIZE_M,
    DEFAULT_TOOL_FRAME,
    CollectorConfig,
    SampleCollector,
)
from handeye_utils import ensure_directory


DEFAULT_OUTPUT_SESSION_DIR = "calibration/eye_in_hand_auto"
DEFAULT_MOVE_GROUP = "ur_manipulator"
DEFAULT_MOVEIT_BASE_LINK = "base_link"
HARDCODED_SOURCE_METADATA = {
    "base_frame": DEFAULT_BASE_FRAME,
    "tool_frame": DEFAULT_TOOL_FRAME,
    "image_topic": DEFAULT_IMAGE_TOPIC,
    "camera_info_topic": DEFAULT_CAMERA_INFO_TOPIC,
    "board": {
        "type": DEFAULT_BOARD_TYPE,
        "cols": DEFAULT_BOARD_COLS,
        "rows": DEFAULT_BOARD_ROWS,
        "square_size_m": DEFAULT_SQUARE_SIZE_M,
        "marker_size_m": DEFAULT_MARKER_SIZE_M,
        "aruco_dict": DEFAULT_ARUCO_DICT,
        "min_charuco_corners": DEFAULT_MIN_CHARUCO_CORNERS,
    },
}
HARDCODED_CALIBRATION_POSES = [
    {
        "index": 0,
        "base_T_tool": {
            "translation_xyz": [
                -0.044405972629315796,
                -0.19275289938549958,
                0.48858159055514294,
            ],
            "quaternion_xyzw": [
                0.9930733042586661,
                0.07223619715013736,
                0.07967537806368316,
                0.04731995689421392,
            ],
        },
    },
    {
        "index": 1,
        "base_T_tool": {
            "translation_xyz": [
                -0.19218259969620247,
                -0.2735611894214903,
                0.40919207790607437,
            ],
            "quaternion_xyzw": [
                0.9891675761454213,
                0.05212403037476149,
                0.13719616657275863,
                -0.002793499207588714,
            ],
        },
    },
    {
        "index": 2,
        "base_T_tool": {
            "translation_xyz": [
                -0.27995491430842645,
                -0.27578998746313893,
                0.31909408619874036,
            ],
            "quaternion_xyzw": [
                0.9691246216738542,
                -0.011536881254993279,
                0.24370063869530148,
                -0.035698273570572654,
            ],
        },
    },
    {
        "index": 3,
        "base_T_tool": {
            "translation_xyz": [
                -0.3018138613091765,
                -0.2889409626299647,
                0.25453036229978004,
            ],
            "quaternion_xyzw": [
                0.9459250297383913,
                0.005625489566205344,
                0.321995409092056,
                -0.03889920957042172,
            ],
        },
    },
    {
        "index": 4,
        "base_T_tool": {
            "translation_xyz": [
                -0.11232424164474107,
                -0.32388383638924345,
                0.40562080005266,
            ],
            "quaternion_xyzw": [
                0.9942625732143688,
                0.016953833517271928,
                0.07047445436340978,
                -0.0786629157633379,
            ],
        },
    },
    {
        "index": 5,
        "base_T_tool": {
            "translation_xyz": [
                -0.15569863058513184,
                -0.3154340819470016,
                0.39848323371589994,
            ],
            "quaternion_xyzw": [
                0.9920205890728002,
                0.0022036445721945754,
                0.11972992230260521,
                -0.03943400197375224,
            ],
        },
    },
    {
        "index": 6,
        "base_T_tool": {
            "translation_xyz": [
                0.07196137132854405,
                -0.3861634006186979,
                0.31738651095047643,
            ],
            "quaternion_xyzw": [
                0.9740934867728611,
                -0.026604098365025867,
                -0.15451336767463741,
                -0.1629715318290784,
            ],
        },
    },
    {
        "index": 7,
        "base_T_tool": {
            "translation_xyz": [
                0.17536749065591764,
                -0.367210011920989,
                0.2744438252802711,
            ],
            "quaternion_xyzw": [
                0.9448183894903662,
                -0.032021384724296977,
                -0.29575659271910193,
                -0.13718921118060368,
            ],
        },
    },
    {
        "index": 8,
        "base_T_tool": {
            "translation_xyz": [
                0.22719841850765832,
                -0.31459513430064867,
                0.3035579304132573,
            ],
            "quaternion_xyzw": [
                0.9398553780722173,
                -0.06451652464837361,
                -0.32968098947010593,
                -0.06180559471469441,
            ],
        },
    },
    {
        "index": 9,
        "base_T_tool": {
            "translation_xyz": [
                0.2168497759906388,
                -0.18005229915932613,
                0.4399545111609958,
            ],
            "quaternion_xyzw": [
                0.973115099442125,
                -0.0956842169470997,
                -0.19553049146329435,
                0.07522872305898887,
            ],
        },
    },
    {
        "index": 10,
        "base_T_tool": {
            "translation_xyz": [
                0.16030795028298558,
                -0.07849913790181669,
                0.5105728591098524,
            ],
            "quaternion_xyzw": [
                0.9697794542266267,
                -0.12314366300822448,
                -0.08283986493204464,
                0.19365176270528323,
            ],
        },
    },
    {
        "index": 11,
        "base_T_tool": {
            "translation_xyz": [
                -0.1814080016103214,
                -0.13757081087361336,
                0.4747299344781232,
            ],
            "quaternion_xyzw": [
                0.9885642242029011,
                -0.05473440337825698,
                0.031114790651722252,
                0.13702842594017497,
            ],
        },
    },
    {
        "index": 12,
        "base_T_tool": {
            "translation_xyz": [
                -0.05133322782033495,
                -0.32301401413174746,
                0.42854671688013257,
            ],
            "quaternion_xyzw": [
                0.9964779072598503,
                0.06362824262926661,
                -0.023077409143668032,
                -0.04950414396953169,
            ],
        },
    },
    {
        "index": 13,
        "base_T_tool": {
            "translation_xyz": [
                -0.0030287048503435943,
                -0.3661543018136365,
                0.3983164666029423,
            ],
            "quaternion_xyzw": [
                0.9724909011174133,
                0.1758270006177662,
                -0.09209631259246404,
                -0.12192039330839066,
            ],
        },
    },
    {
        "index": 14,
        "base_T_tool": {
            "translation_xyz": [
                -0.15205417107874586,
                -0.3531775997991457,
                0.40362140774420197,
            ],
            "quaternion_xyzw": [
                0.9466003272301693,
                0.3214288833431627,
                -0.005877362785259321,
                -0.024428467966586227,
            ],
        },
    },
    {
        "index": 15,
        "base_T_tool": {
            "translation_xyz": [
                -0.22384194590639928,
                -0.23664081010803703,
                0.4293458906515901,
            ],
            "quaternion_xyzw": [
                0.9595548443365645,
                0.23599769929062786,
                0.11265963042808473,
                0.10424679521006579,
            ],
        },
    },
    {
        "index": 16,
        "base_T_tool": {
            "translation_xyz": [
                -0.27347297944423177,
                -0.25212091161809197,
                0.3381270055223291,
            ],
            "quaternion_xyzw": [
                0.9504311731954767,
                0.03325247019388602,
                0.30755021637057756,
                0.03142805521640408,
            ],
        },
    },
    {
        "index": 17,
        "base_T_tool": {
            "translation_xyz": [
                -0.31079160341474205,
                -0.26593981652094906,
                0.26332457403955284,
            ],
            "quaternion_xyzw": [
                0.9274417509167515,
                0.01854450612188005,
                0.3731637242951341,
                0.016022946650068128,
            ],
        },
    },
    {
        "index": 18,
        "base_T_tool": {
            "translation_xyz": [
                0.14952165943657578,
                -0.3634120199731842,
                0.3518853306454316,
            ],
            "quaternion_xyzw": [
                0.9543507633320203,
                0.11669781240541538,
                -0.24988767769869114,
                -0.11468386827159234,
            ],
        },
    },
    {
        "index": 19,
        "base_T_tool": {
            "translation_xyz": [
                0.22209074198066314,
                -0.33129081788707443,
                0.33835743686926417,
            ],
            "quaternion_xyzw": [
                0.9063507639291867,
                0.26758233831750394,
                -0.3053555814867746,
                -0.11698698132865282,
            ],
        },
    },
]


class AutoCalibrationCollector(SampleCollector):
    def __init__(
        self,
        *,
        config: CollectorConfig,
        move_group: str,
        moveit_base_link: str,
        move_target_link: str,
        velocity_scale: float,
        acceleration_scale: float,
        moveit_wait_sec: float,
    ):
        super().__init__(config)
        self.move_target_link = move_target_link
        self.callback_group = ReentrantCallbackGroup()
        self._moveit = MoveIt2(
            node=self,
            joint_names=ur.joint_names(prefix=""),
            base_link_name=moveit_base_link,
            end_effector_name=move_target_link,
            group_name=move_group,
            callback_group=self.callback_group,
            use_move_group_action=True,
        )
        self._moveit.max_velocity = velocity_scale
        self._moveit.max_acceleration = acceleration_scale
        self._plan_client = self.create_client(
            srv_type=GetMotionPlan,
            srv_name="plan_kinematic_path",
            callback_group=self.callback_group,
        )
        self.get_logger().info(
            f"Waiting up to {moveit_wait_sec:.1f} seconds for MoveIt planning."
        )
        if not self._plan_client.wait_for_service(timeout_sec=moveit_wait_sec):
            self.get_logger().warn("MoveIt planning is not available yet.")

        move_action_client = getattr(
            self._moveit, "_MoveIt2__move_action_client", None
        )
        if move_action_client is not None:
            self.get_logger().info(
                f"Waiting up to {moveit_wait_sec:.1f} seconds for the MoveIt action server."
            )
            if not move_action_client.wait_for_server(timeout_sec=moveit_wait_sec):
                self.get_logger().warn("The MoveIt action server is not available yet.")

    def wait_for_camera(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.camera_matrix is not None and self.recent_frames:
                return True
        return False

    def moveit_ready(self) -> bool:
        if not self._plan_client.service_is_ready():
            self.get_logger().warn("MoveIt planning is not available.")
            return False

        move_action_client = getattr(
            self._moveit, "_MoveIt2__move_action_client", None
        )
        if (
            move_action_client is not None
            and not move_action_client.server_is_ready()
        ):
            self.get_logger().warn("The MoveIt action server is not available.")
            return False

        return True

    def is_tf_lookup_acceptable(self, tf_lookup: dict | None) -> tuple[bool, str]:
        if (
            tf_lookup is not None
            and tf_lookup["used_latest_tf"]
            and self.config.allow_latest_tf_fallback
        ):
            self.get_logger().warn(
                "Using latest TF because exact image time TF is unavailable. "
                "This is acceptable for auto capture only after the robot has settled."
            )
            return True, ""

        return super().is_tf_lookup_acceptable(tf_lookup)

    def move_to_saved_pose(
        self,
        *,
        pose: dict,
        frame_id: str,
        position_tolerance: float,
        orientation_tolerance: float,
        timeout_sec: float,
    ) -> bool:
        if not self.moveit_ready():
            return False

        position = pose["translation_xyz"]
        quat_xyzw = pose["quaternion_xyzw"]
        self.get_logger().info(
            "Planning calibration move to "
            f"{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}. "
            f"Frame is {frame_id}. Target link is {self.move_target_link}."
        )

        try:
            self._moveit.move_to_pose(
                position=position,
                quat_xyzw=quat_xyzw,
                target_link=self.move_target_link,
                frame_id=frame_id,
                tolerance_position=position_tolerance,
                tolerance_orientation=orientation_tolerance,
                cartesian=False,
            )
        except Exception as exc:
            self.get_logger().warn(f"MoveIt rejected the pose request. {exc}")
            return False

        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._moveit.query_state() == MoveIt2State.IDLE:
                return bool(self._moveit.motion_suceeded)

        self.get_logger().warn(
            f"Timed out waiting for calibration move after {timeout_sec:.1f} seconds."
        )
        if self._moveit.query_state() == MoveIt2State.EXECUTING:
            self._moveit.cancel_execution()
        return False

    def wait_for_detected_target(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if any(
                frame["pose_object_points"] is not None
                and frame["pose_image_points"] is not None
                for frame in self.recent_frames
            ):
                return True
        return False

    def capture_after_settle(
        self,
        *,
        settle_sec: float,
        capture_timeout_sec: float,
    ) -> bool:
        self.recent_frames.clear()
        settle_until = time.monotonic() + settle_sec
        while rclpy.ok() and time.monotonic() < settle_until:
            rclpy.spin_once(self, timeout_sec=0.05)

        self.recent_frames.clear()
        if not self.wait_for_detected_target(capture_timeout_sec):
            self.get_logger().warn(
                f"No calibration target was detected within {capture_timeout_sec:.1f} seconds."
            )
            return False

        sample_count_before = len(self.samples)
        self.save_current_sample()
        return len(self.samples) > sample_count_before


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically move through hardcoded eye-in-hand poses and capture "
            "new ChArUco/checkerboard samples."
        )
    )
    parser.add_argument(
        "--session-dir",
        default=DEFAULT_OUTPUT_SESSION_DIR,
        help="Output session for the newly captured samples.",
    )
    parser.add_argument("--image-topic", default="")
    parser.add_argument("--camera-info-topic", default="")
    parser.add_argument("--base-frame", default="")
    parser.add_argument("--tool-frame", default="")
    parser.add_argument(
        "--move-frame",
        default="",
        help="Frame used for MoveIt pose goals. Defaults to the hardcoded base frame.",
    )
    parser.add_argument(
        "--move-target-link",
        default="",
        help="Robot link to move to each hardcoded pose. Defaults to the hardcoded tool frame.",
    )
    parser.add_argument("--move-group", default=DEFAULT_MOVE_GROUP)
    parser.add_argument("--moveit-base-link", default=DEFAULT_MOVEIT_BASE_LINK)
    parser.add_argument("--max-poses", type=int, default=0)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--capture-timeout-sec", type=float, default=3.0)
    parser.add_argument("--motion-timeout-sec", type=float, default=45.0)
    parser.add_argument("--startup-timeout-sec", type=float, default=10.0)
    parser.add_argument("--moveit-wait-sec", type=float, default=5.0)
    parser.add_argument("--velocity-scale", type=float, default=0.2)
    parser.add_argument("--acceleration-scale", type=float, default=0.2)
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance", type=float, default=0.05)
    parser.add_argument("--max-tf-delta-ms", type=float, default=50.0)
    parser.add_argument(
        "--require-exact-tf",
        action="store_true",
        help=(
            "Reject captures unless TF is available at the exact image timestamp. "
            "By default auto capture accepts latest TF after the robot settles."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Start robot motion without the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the hardcoded poses and exit without moving the robot.",
    )
    return parser.parse_args()


def build_config(args, source_payload: dict) -> CollectorConfig:
    metadata = source_payload.get("metadata", {})
    board = metadata.get("board", {})
    board_type = str(board.get("type", DEFAULT_BOARD_TYPE))
    aruco_dict = str(board.get("aruco_dict", DEFAULT_ARUCO_DICT))
    if board_type == "charuco" and aruco_dict not in ARUCO_DICTIONARIES:
        raise ValueError(f"Unsupported ArUco dictionary in source session: {aruco_dict}")

    return CollectorConfig(
        image_topic=args.image_topic or metadata.get("image_topic", DEFAULT_IMAGE_TOPIC),
        camera_info_topic=(
            args.camera_info_topic
            or metadata.get("camera_info_topic", DEFAULT_CAMERA_INFO_TOPIC)
        ),
        base_frame=args.base_frame or metadata.get("base_frame", DEFAULT_BASE_FRAME),
        tool_frame=args.tool_frame or metadata.get("tool_frame", DEFAULT_TOOL_FRAME),
        board_type=board_type,
        board_cols=int(board.get("cols", DEFAULT_BOARD_COLS)),
        board_rows=int(board.get("rows", DEFAULT_BOARD_ROWS)),
        square_size_m=float(board.get("square_size_m", DEFAULT_SQUARE_SIZE_M)),
        marker_size_m=(
            float(board.get("marker_size_m", DEFAULT_MARKER_SIZE_M))
            if board_type == "charuco"
            else None
        ),
        aruco_dict=aruco_dict,
        min_charuco_corners=max(
            4, int(board.get("min_charuco_corners", DEFAULT_MIN_CHARUCO_CORNERS))
        ),
        session_dir=Path(args.session_dir),
        preview_scale=1.0,
        max_tf_delta_ms=max(0.0, args.max_tf_delta_ms),
        allow_latest_tf_fallback=not bool(args.require_exact_tf),
    )


def confirm_motion(
    *,
    args,
    poses: list[dict],
    output_session_dir: Path,
    move_frame: str,
    move_target_link: str,
):
    print("Using the hardcoded ChArUco calibration poses.")
    print(f"Saving samples in {output_session_dir}.")
    print(f"Pose count is {len(poses)}.")
    print(f"Move frame is {move_frame}.")
    print(f"Move target link is {move_target_link}.")
    print(f"Velocity and acceleration scale are {args.velocity_scale}, {args.acceleration_scale}.")
    print("Existing samples in the output session will be replaced.")

    if args.dry_run:
        for sample in poses:
            pose = sample["base_T_tool"]
            xyz = pose["translation_xyz"]
            print(
                f"Sample {sample.get('index')}. "
                f"Position {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}."
            )
        return

    if args.yes:
        return

    print()
    print("The robot will move automatically through these saved poses.")
    print("Keep the emergency stop reachable and make sure the workspace is clear.")
    answer = input("Type yes to start. ").strip().lower()
    if answer != "yes":
        raise RuntimeError("Stopped before robot motion.")


def clear_output_samples(session_dir: Path) -> int:
    removed_count = 0

    for path in (
        session_dir / "samples.json",
        session_dir / "samples.json.bak",
        session_dir / "samples_pruned.json",
        session_dir / "handeye_result.json",
    ):
        if path.exists():
            path.unlink()
            removed_count += 1

    images_dir = session_dir / "images"
    if images_dir.exists():
        for path in images_dir.glob("sample_*"):
            if path.is_file():
                path.unlink()
                removed_count += 1

    return removed_count


def main():
    args = parse_args()
    source_payload = {"metadata": HARDCODED_SOURCE_METADATA}
    source_samples = list(HARDCODED_CALIBRATION_POSES)
    if args.max_poses > 0:
        source_samples = source_samples[: args.max_poses]

    config = build_config(args, source_payload)
    move_frame = args.move_frame or source_payload.get("metadata", {}).get(
        "base_frame", DEFAULT_BASE_FRAME
    )
    move_target_link = args.move_target_link or source_payload.get("metadata", {}).get(
        "tool_frame", DEFAULT_TOOL_FRAME
    )

    confirm_motion(
        args=args,
        poses=source_samples,
        output_session_dir=config.session_dir,
        move_frame=move_frame,
        move_target_link=move_target_link,
    )
    if args.dry_run:
        return

    ensure_directory(config.session_dir)
    removed_count = clear_output_samples(config.session_dir)
    if removed_count > 0:
        print(f"Cleared {removed_count} old calibration sample files.")

    rclpy.init(args=None)
    node = AutoCalibrationCollector(
        config=config,
        move_group=args.move_group,
        moveit_base_link=args.moveit_base_link,
        move_target_link=move_target_link,
        velocity_scale=max(0.01, min(1.0, args.velocity_scale)),
        acceleration_scale=max(0.01, min(1.0, args.acceleration_scale)),
        moveit_wait_sec=max(0.0, args.moveit_wait_sec),
    )

    saved_count_before = len(node.samples)
    try:
        if not node.wait_for_camera(max(0.1, args.startup_timeout_sec)):
            raise RuntimeError(
                "Timed out waiting for camera image and camera info topics."
            )

        for pose_number, sample in enumerate(source_samples, start=1):
            node.get_logger().info(
                f"Pose {pose_number} of {len(source_samples)}. "
                f"Source index {sample.get('index')}."
            )
            moved = node.move_to_saved_pose(
                pose=sample["base_T_tool"],
                frame_id=move_frame,
                position_tolerance=max(0.0, args.position_tolerance),
                orientation_tolerance=max(0.0, args.orientation_tolerance),
                timeout_sec=max(1.0, args.motion_timeout_sec),
            )
            if not moved:
                node.get_logger().warn("Skipping capture because the move failed.")
                continue

            if node.capture_after_settle(
                settle_sec=max(0.0, args.settle_sec),
                capture_timeout_sec=max(0.1, args.capture_timeout_sec),
            ):
                node.get_logger().info("Captured sample at this pose.")
            else:
                node.get_logger().warn("Skipping this pose because capture failed.")
    finally:
        saved_now = len(node.samples) - saved_count_before
        node.get_logger().info(
            f"Automatic calibration capture is done. Saved {saved_now} new samples."
        )
        node.print_tf_lookup_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
