#!/usr/bin/env python3
from __future__ import annotations

import ast
import math
import socket
import time
from contextlib import contextmanager
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from moveit_msgs.srv import GetMotionPlan
from pymoveit2 import MoveIt2
from pymoveit2.moveit2 import MoveIt2State
from pymoveit2.robots import ur
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2DArray


class PickMoveItExecutorNode(Node):
    GROUP_NAME = "ur_manipulator"
    BASE_LINK_NAME = "base_link"
    END_EFFECTOR_NAME = "gripper_tcp"
    TARGET_LINK = "gripper_tcp"
    ELBOW_HEIGHT_LINK_NAME = "forearm_link"
    JOINT_TOLERANCE = 0.01
    MAX_POSE_AGE_SEC = 2.0
    POSITION_TOLERANCE = 0.005
    ORIENTATION_TOLERANCE = 0.05
    LINEAR_GRASP_PIPELINE_ID = "pilz_industrial_motion_planner"
    LINEAR_GRASP_PLANNER_ID = "LIN"
    JOINT_POSITION_LIMITS_RAD = {
        "shoulder_pan_joint": (-2.0 * math.pi, 2.0 * math.pi),
        "shoulder_lift_joint": (-2.0 * math.pi, 2.0 * math.pi),
        "elbow_joint": (-2.0 * math.pi, 2.0 * math.pi),
        "wrist_1_joint": (-2.0 * math.pi, 2.0 * math.pi),
        "wrist_2_joint": (-2.0 * math.pi, 2.0 * math.pi),
        "wrist_3_joint": (-2.0 * math.pi, 2.0 * math.pi),
    }
    MAX_REASONABLE_JOINT_MOVE_RAD = math.radians(180.0)
    MOVEIT_SUCCESS = 1
    MOVEIT_WAIT_SECONDS = 5.0
    IK_WAIT_SECONDS = 3.0
    EXECUTION_TIMEOUT_SEC = 30.0
    EXECUTION_POLL_INTERVAL_SEC = 0.05
    DEFAULT_ROBOT_IP = "192.168.0.100"
    ROBOT_SCRIPT_PORT = 30002
    ROBOT_SCRIPT_TIMEOUT_SEC = 2.0
    GRIPPER_SETTLE_SEC = 1.0
    SORT_DROP_SLOTS_PER_CLASS = 2

    def __init__(self):
        super().__init__("pick_moveit_executor_node")

        self.declare_parameter("approach_topic", "/pick_approach_pose")
        self.declare_parameter("grasp_topic", "/pick_grasp_pose")
        self.declare_parameter("target_class_topic", "/pick_target_class")
        self.declare_parameter("yolo_detection_topic", "/yolo/detections")
        self.declare_parameter("status_topic", "/pick_execution_status")
        self.declare_parameter("motion_active_topic", "/pick_motion_active")
        self.declare_parameter("approach_to_grasp_wait_sec", 1.0)
        self.declare_parameter("approach_fallback_enabled", True)
        self.declare_parameter("approach_fallback_xy_step", 0.02)
        self.declare_parameter("approach_fallback_xy_levels", 2)
        self.declare_parameter("approach_fallback_z_step", 0.01)
        self.declare_parameter("approach_fallback_z_levels", 2)
        self.declare_parameter("pre_grasp_clearance_z", 0.05)
        self.declare_parameter("grasp_velocity_scaling", 0.15)
        self.declare_parameter("grasp_acceleration_scaling", 0.05)
        self.declare_parameter("prefer_elbow_up_ik", True)
        self.declare_parameter("elbow_up_seed_deg", 90.0)
        self.declare_parameter(
            "ready_joint_positions_deg",
            [-90.0, -90.0, 0.0, -180.0, 90.0, 180.0],
        )
        self.declare_parameter(
            "dice_drop_joint_positions_deg",
            [-80.0, -120.0, 0.0, -140.0, 90.0, 190.0],
        )
        self.declare_parameter(
            "dice_repick_joint_positions_deg",
            [-80.0, -105.0, 0.0, -163.0, 90.0, 190.0],
        )
        self.declare_parameter("dice_repick_wait_sec", 2.0)
        self.declare_parameter(
            "search_start_joint_positions_deg",
            [-80.0, -105.0, 0.0, -163.0, 90.0, 190.0],
        )
        self.declare_parameter(
            "search_look_offsets_deg",
            "[[0.0, 0.0, 0.0, 5.0, 12.0, 0.0], "
            "[0.0, 0.0, 0.0, 5.0, -10.0, 0.0], "
            "[0.0, -9.0, 0.0, -17.0, -3.0, 8.0], "
            "[0.0, 0.0, 0.0, -19.0, 20.0, -6.0]]",
        )
        self.declare_parameter("search_joint_positions_deg", "[]")
        self.declare_parameter("search_pose_wait_sec", 1.0)
        self.declare_parameter(
            "sort_drop_classes",
            "['big_stick', 'big_cube', 'small_stick', 'small_cube']",
        )
        self.declare_parameter("sort_drop_frame", "base")
        self.declare_parameter("sort_drop_base_xyz_m", [-0.36363, 0.03965, -0.00850])
        self.declare_parameter("sort_drop_base_rotvec_rad", [2.199, -2.213, 0.004])
        self.declare_parameter("sort_drop_slot_offset_y", 0.05)
        self.declare_parameter("sort_drop_table_top_z", -0.0165)
        self.declare_parameter("sort_drop_release_clearance_z", 0.015)
        self.declare_parameter("sort_drop_approach_clearance_z", 0.10)
        self.declare_parameter("robot_ip", self.DEFAULT_ROBOT_IP)

        self.approach_topic = str(self.get_parameter("approach_topic").value)
        self.grasp_topic = str(self.get_parameter("grasp_topic").value)
        self.target_class_topic = str(self.get_parameter("target_class_topic").value)
        self.yolo_detection_topic = str(
            self.get_parameter("yolo_detection_topic").value
        )
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.motion_active_topic = str(self.get_parameter("motion_active_topic").value)
        self.approach_to_grasp_wait_sec = max(
            float(self.get_parameter("approach_to_grasp_wait_sec").value), 0.0
        )
        self.approach_fallback_enabled = bool(
            self.get_parameter("approach_fallback_enabled").value
        )
        self.approach_fallback_xy_step = max(
            float(self.get_parameter("approach_fallback_xy_step").value), 0.0
        )
        self.approach_fallback_xy_levels = max(
            int(self.get_parameter("approach_fallback_xy_levels").value), 0
        )
        self.approach_fallback_z_step = max(
            float(self.get_parameter("approach_fallback_z_step").value), 0.0
        )
        self.approach_fallback_z_levels = max(
            int(self.get_parameter("approach_fallback_z_levels").value), 0
        )
        self.pre_grasp_clearance_z = max(
            float(self.get_parameter("pre_grasp_clearance_z").value), 0.0
        )
        self.grasp_velocity_scaling = self._clamp_speed_scaling(
            float(self.get_parameter("grasp_velocity_scaling").value)
        )
        self.grasp_acceleration_scaling = self._clamp_speed_scaling(
            float(self.get_parameter("grasp_acceleration_scaling").value)
        )
        self.prefer_elbow_up_ik = bool(
            self.get_parameter("prefer_elbow_up_ik").value
        )
        self.elbow_up_seed_rad = math.radians(
            float(self.get_parameter("elbow_up_seed_deg").value)
        )
        self.robot_ip = str(self.get_parameter("robot_ip").value).strip()
        self.ready_joint_positions_deg = self._parse_joint_positions_deg(
            self.get_parameter("ready_joint_positions_deg").value,
            "ready_joint_positions_deg",
        )
        self.ready_joint_positions_rad = [
            math.radians(value) for value in self.ready_joint_positions_deg
        ]
        self.dice_drop_joint_positions_deg = self._parse_joint_positions_deg(
            self.get_parameter("dice_drop_joint_positions_deg").value,
            "dice_drop_joint_positions_deg",
        )
        self.dice_drop_joint_positions_rad = [
            math.radians(value) for value in self.dice_drop_joint_positions_deg
        ]
        self.dice_repick_joint_positions_deg = self._parse_joint_positions_deg(
            self.get_parameter("dice_repick_joint_positions_deg").value,
            "dice_repick_joint_positions_deg",
        )
        self.dice_repick_joint_positions_rad = [
            math.radians(value) for value in self.dice_repick_joint_positions_deg
        ]
        self.dice_repick_wait_sec = max(
            float(self.get_parameter("dice_repick_wait_sec").value), 0.0
        )
        self.search_start_joint_positions_deg = self._parse_joint_positions_deg(
            self.get_parameter("search_start_joint_positions_deg").value,
            "search_start_joint_positions_deg",
        )
        self.search_start_joint_positions_rad = [
            math.radians(value) for value in self.search_start_joint_positions_deg
        ]
        self.search_look_offsets_deg = self._parse_joint_position_sets_deg(
            self.get_parameter("search_look_offsets_deg").value,
            "search_look_offsets_deg",
        )
        self.search_joint_positions_deg = self._parse_joint_position_sets_deg(
            self.get_parameter("search_joint_positions_deg").value,
            "search_joint_positions_deg",
        )
        self.search_joint_positions_rad = [
            [math.radians(value) for value in joint_positions]
            for joint_positions in self.search_joint_positions_deg
        ]
        self.search_pose_wait_sec = max(
            float(self.get_parameter("search_pose_wait_sec").value), 0.0
        )
        self.sort_drop_classes = [
            self._normalize_class_name(value)
            for value in self._parse_string_list(
                self.get_parameter("sort_drop_classes").value,
                "sort_drop_classes",
            )
        ]
        self.sort_drop_frame = str(self.get_parameter("sort_drop_frame").value).strip()
        if not self.sort_drop_frame:
            self.sort_drop_frame = self.BASE_LINK_NAME
        self.sort_drop_base_xyz_m = self._parse_float_list(
            self.get_parameter("sort_drop_base_xyz_m").value,
            "sort_drop_base_xyz_m",
            3,
        )
        self.sort_drop_base_rotvec_rad = self._parse_float_list(
            self.get_parameter("sort_drop_base_rotvec_rad").value,
            "sort_drop_base_rotvec_rad",
            3,
        )
        self.sort_drop_slot_offset_y = max(
            float(self.get_parameter("sort_drop_slot_offset_y").value), 0.0
        )
        self.sort_drop_table_top_z = float(
            self.get_parameter("sort_drop_table_top_z").value
        )
        self.sort_drop_release_clearance_z = max(
            float(self.get_parameter("sort_drop_release_clearance_z").value), 0.0
        )
        self.sort_drop_approach_clearance_z = max(
            float(self.get_parameter("sort_drop_approach_clearance_z").value), 0.0
        )

        self.callback_group = ReentrantCallbackGroup()
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.motion_active_pub = self.create_publisher(
            Bool, self.motion_active_topic, 10
        )

        self.latest_approach_pose: Optional[PoseStamped] = None
        self.latest_grasp_pose: Optional[PoseStamped] = None
        self.latest_approach_received_ns: Optional[int] = None
        self.latest_grasp_received_ns: Optional[int] = None
        self.last_completed_grasp_pose: Optional[PoseStamped] = None
        self.last_sort_drop_lift_pose: Optional[PoseStamped] = None
        self.latest_target_class = ""
        self.latest_target_class_received_ns: Optional[int] = None
        self.latest_yolo_detection_received_ns: Optional[int] = None
        self.latest_yolo_detection_count = 0
        self.sort_drop_counts = {class_name: 0 for class_name in self.sort_drop_classes}
        self._motion_active_depth = 0
        self._dice_test_running = False
        self._dice_test_stop_requested = False
        self._sorting_running = False
        self._sorting_stop_requested = False

        self.create_subscription(
            PoseStamped,
            self.approach_topic,
            self._on_approach_pose,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PoseStamped,
            self.grasp_topic,
            self._on_grasp_pose,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            String,
            self.target_class_topic,
            self._on_target_class,
            10,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            Detection2DArray,
            self.yolo_detection_topic,
            self._on_yolo_detections,
            10,
            callback_group=self.callback_group,
        )

        self.create_service(
            Trigger,
            "~/execute_approach",
            self._handle_execute_approach,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/execute_centered_approach",
            self._handle_execute_centered_approach,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/execute_grasp",
            self._handle_execute_grasp,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/execute_pick",
            self._handle_execute_pick,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/search_workspace",
            self._handle_search_workspace,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/run_dice_test",
            self._handle_run_dice_test,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/stop_dice_test",
            self._handle_stop_dice_test,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/run_sorting",
            self._handle_run_sorting,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/stop_sorting",
            self._handle_stop_sorting,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/move_to_start_pose",
            self._handle_move_to_start_pose,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/open_gripper",
            self._handle_open_gripper,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/close_gripper",
            self._handle_close_gripper,
            callback_group=self.callback_group,
        )

        self._moveit = MoveIt2(
            node=self,
            joint_names=ur.joint_names(prefix=""),
            base_link_name=self.BASE_LINK_NAME,
            end_effector_name=self.END_EFFECTOR_NAME,
            group_name=self.GROUP_NAME,
            callback_group=self.callback_group,
            use_move_group_action=True,
        )
        self._moveit.max_velocity = 1.0
        self._moveit.max_acceleration = 1.0
        self._moveit.pipeline_id = "ompl"
        self._moveit.planner_id = "RRTConnect"
        self._moveit.allowed_planning_time = 3.0
        self._moveit.num_planning_attempts = 10
        self._moveit._MoveIt2__move_action_goal.planning_options.replan = True
        self._moveit._MoveIt2__move_action_goal.planning_options.replan_attempts = 10

        self._plan_client = self.create_client(
            srv_type=GetMotionPlan,
            srv_name="plan_kinematic_path",
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            f"Waiting for MoveIt, up to {self.MOVEIT_WAIT_SECONDS:.1f}s."
        )
        self._plan_client.wait_for_service(timeout_sec=self.MOVEIT_WAIT_SECONDS)

        self.get_logger().info(f"Reading approach poses from {self.approach_topic}.")
        self.get_logger().info(f"Reading grasp poses from {self.grasp_topic}.")
        self.get_logger().info(f"Reading target classes from {self.target_class_topic}.")
        self.get_logger().info(
            f"Reading YOLO detections from {self.yolo_detection_topic}."
        )
        self.get_logger().info(f"Publishing robot status on {self.status_topic}.")
        self.get_logger().info(
            f"Publishing the motion active flag on {self.motion_active_topic}."
        )
        self.get_logger().info("Pick services are ready.")
        self.get_logger().info(
            f"Using MoveIt group {self.GROUP_NAME} from {self.BASE_LINK_NAME} "
            f"to {self.TARGET_LINK}."
        )
        self.get_logger().info(
            f"Ready pose joints in degrees are {self.ready_joint_positions_deg}."
        )
        self.get_logger().info(
            f"Dice drop joints in degrees are {self.dice_drop_joint_positions_deg}."
        )
        self.get_logger().info(
            f"Dice re pick joints in degrees are {self.dice_repick_joint_positions_deg}."
        )
        self.get_logger().info(f"Dice re pick wait is {self.dice_repick_wait_sec:.2f} s.")
        self.get_logger().info(
            f"Search start joints in degrees are {self.search_start_joint_positions_deg}."
        )
        self.get_logger().info(
            f"Search has {len(self.search_look_offsets_deg)} look offsets, "
            f"{len(self.search_joint_positions_deg)} extra poses, "
            f"and waits {self.search_pose_wait_sec:.2f} s at each pose."
        )
        self.get_logger().info(
            "Sorting classes are "
            f"{', '.join(self.sort_drop_classes)}. "
            f"Drop frame is {self.sort_drop_frame}. "
            f"Drop point is {self.sort_drop_base_xyz_m[0]:.3f}, "
            f"{self.sort_drop_base_xyz_m[1]:.3f}, "
            f"{self._sort_drop_release_z():.3f} m. "
            f"Table is {self.sort_drop_table_top_z:.3f} m. "
            f"Release clearance is {self.sort_drop_release_clearance_z:.3f} m. "
            f"Slot spacing is {self.sort_drop_slot_offset_y:.3f} m. "
            f"Approach clearance is {self.sort_drop_approach_clearance_z:.3f} m."
        )
        self.get_logger().info(
            f"Approach to grasp wait is {self.approach_to_grasp_wait_sec:.2f} s."
        )
        self.get_logger().info(
            f"Pre grasp clearance is {self.pre_grasp_clearance_z:.3f} m."
        )
        self.get_logger().info(
            "Grasp moves are slowed down. "
            f"velocity {self.grasp_velocity_scaling:.2f}, "
            f"acceleration {self.grasp_acceleration_scaling:.2f}."
        )
        self.get_logger().info(
            "Elbow up IK is "
            f"{'on' if self.prefer_elbow_up_ik else 'off'}, "
            f"using elbow seeds around {math.degrees(self.elbow_up_seed_rad):.1f} degrees "
            f"and picking the highest {self.ELBOW_HEIGHT_LINK_NAME}."
        )
        self.get_logger().info(
            "Nearby approach search is "
            f"{'on' if self.approach_fallback_enabled else 'off'}, "
            f"xy step {self.approach_fallback_xy_step:.3f} m, "
            f"z step {self.approach_fallback_z_step:.3f} m."
        )
        self.get_logger().info(f"Sending gripper scripts to {self.robot_ip} on port 30002.")

    def _parse_joint_positions_deg(self, value, parameter_name: str) -> list[float]:
        if isinstance(value, str):
            try:
                parsed_value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"{parameter_name} must be a list string like "
                    '"[-90.0, -90.0, 0.0, -180.0, 90.0, 180.0]".'
                ) from exc
        else:
            parsed_value = value

        if not isinstance(parsed_value, (list, tuple)):
            raise ValueError(f"{parameter_name} must contain six joint values.")

        joint_positions = [float(item) for item in parsed_value]
        expected_joint_count = len(ur.joint_names(prefix=""))
        if len(joint_positions) != expected_joint_count:
            raise ValueError(
                f"{parameter_name} must contain "
                f"{expected_joint_count} values in UR order "
                "[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]."
            )

        return joint_positions

    def _parse_joint_position_sets_deg(
        self, value, parameter_name: str
    ) -> list[list[float]]:
        if isinstance(value, str):
            if not value.strip():
                return []
            try:
                parsed_value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"{parameter_name} must be a list string like "
                    '"[[-80.0, -105.0, 0.0, -163.0, 90.0, 190.0]]".'
                ) from exc
        else:
            parsed_value = value

        if parsed_value is None or parsed_value == []:
            return []
        if not isinstance(parsed_value, (list, tuple)):
            raise ValueError(f"{parameter_name} must contain joint target lists.")

        expected_joint_count = len(ur.joint_names(prefix=""))
        if len(parsed_value) == expected_joint_count and all(
            isinstance(item, (int, float)) for item in parsed_value
        ):
            return [
                self._parse_joint_positions_deg(
                    parsed_value,
                    parameter_name,
                )
            ]

        joint_position_sets: list[list[float]] = []
        for index, item in enumerate(parsed_value):
            try:
                joint_position_sets.append(
                    self._parse_joint_positions_deg(
                        item,
                        f"{parameter_name}[{index}]",
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{parameter_name}[{index}] must contain "
                    f"{expected_joint_count} joint values."
                ) from exc

        return joint_position_sets

    @staticmethod
    def _normalize_class_name(value: str) -> str:
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _parse_string_list(value, parameter_name: str) -> list[str]:
        if isinstance(value, str):
            try:
                parsed_value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"{parameter_name} must be a list string like "
                    "\"['big_stick', 'big_cube']\"."
                ) from exc
        else:
            parsed_value = value

        if not isinstance(parsed_value, (list, tuple)) or not parsed_value:
            raise ValueError(f"{parameter_name} must contain at least one class name.")
        return [str(item) for item in parsed_value]

    @staticmethod
    def _parse_float_list(
        value, parameter_name: str, expected_count: int
    ) -> list[float]:
        if isinstance(value, str):
            try:
                parsed_value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    f"{parameter_name} must be a list of {expected_count} numbers."
                ) from exc
        else:
            parsed_value = value

        if not isinstance(parsed_value, (list, tuple)):
            raise ValueError(
                f"{parameter_name} must be a list of {expected_count} numbers."
            )

        parsed_list = [float(item) for item in parsed_value]
        if len(parsed_list) != expected_count:
            raise ValueError(
                f"{parameter_name} must contain exactly {expected_count} numbers."
            )
        return parsed_list

    def _on_approach_pose(self, msg: PoseStamped) -> None:
        if self._motion_active_depth > 0:
            return
        self.latest_approach_pose = msg
        self.latest_approach_received_ns = self.get_clock().now().nanoseconds

    def _on_grasp_pose(self, msg: PoseStamped) -> None:
        if self._motion_active_depth > 0:
            return
        self.latest_grasp_pose = msg
        self.latest_grasp_received_ns = self.get_clock().now().nanoseconds

    def _on_target_class(self, msg: String) -> None:
        self.latest_target_class = self._normalize_class_name(msg.data)
        self.latest_target_class_received_ns = self.get_clock().now().nanoseconds

    def _on_yolo_detections(self, msg: Detection2DArray) -> None:
        if self._motion_active_depth > 0:
            return
        detection_count = len(msg.detections)
        if detection_count <= 0:
            return
        self.latest_yolo_detection_count = detection_count
        self.latest_yolo_detection_received_ns = self.get_clock().now().nanoseconds

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def _publish_motion_active(self, active: bool) -> None:
        msg = Bool()
        msg.data = bool(active)
        self.motion_active_pub.publish(msg)

    @staticmethod
    def _clamp_speed_scaling(value: float) -> float:
        return min(max(value, 0.01), 1.0)

    @contextmanager
    def _motion_active_guard(self):
        self._motion_active_depth += 1
        if self._motion_active_depth == 1:
            self._publish_motion_active(True)
        try:
            yield
        finally:
            self._motion_active_depth = max(self._motion_active_depth - 1, 0)
            if self._motion_active_depth == 0:
                self._publish_motion_active(False)

    @contextmanager
    def _moveit_speed_guard(self, velocity_scaling: float, acceleration_scaling: float):
        old_velocity = self._moveit.max_velocity
        old_acceleration = self._moveit.max_acceleration
        self._moveit.max_velocity = velocity_scaling
        self._moveit.max_acceleration = acceleration_scaling
        try:
            yield
        finally:
            self._moveit.max_velocity = old_velocity
            self._moveit.max_acceleration = old_acceleration

    @contextmanager
    def _moveit_planner_guard(self, pipeline_id: str, planner_id: str):
        old_pipeline_id = self._moveit.pipeline_id
        old_planner_id = self._moveit.planner_id
        self._moveit.pipeline_id = pipeline_id
        self._moveit.planner_id = planner_id
        try:
            yield
        finally:
            self._moveit.pipeline_id = old_pipeline_id
            self._moveit.planner_id = old_planner_id

    def _moveit_ready(self) -> bool:
        ready = self._plan_client.service_is_ready()
        if not ready:
            self.get_logger().warn(
                "MoveIt is not ready yet. Start move group first, or launch with MoveIt enabled."
            )
        return ready

    def _wait_for_motion_completion(self, label: str) -> bool:
        deadline = time.monotonic() + self.EXECUTION_TIMEOUT_SEC

        while time.monotonic() < deadline:
            state = self._moveit.query_state()
            if state == MoveIt2State.IDLE:
                return bool(self._moveit.motion_suceeded)
            time.sleep(self.EXECUTION_POLL_INTERVAL_SEC)

        self.get_logger().warn(
            f"Timed out waiting for {label} to finish after "
            f"{self.EXECUTION_TIMEOUT_SEC:.1f} seconds."
        )
        return False

    def _current_joint_positions(self, joint_names: list[str]) -> Optional[list[float]]:
        joint_state = self._moveit.joint_state
        if joint_state is None:
            self.get_logger().warn("I do not have a joint state yet. Using the requested angles.")
            return None

        positions_by_name = {
            name: float(position)
            for name, position in zip(joint_state.name, joint_state.position)
        }
        missing_joint_names = [
            name for name in joint_names if name not in positions_by_name
        ]
        if missing_joint_names:
            self.get_logger().warn(
                f"I am missing joint state for {', '.join(missing_joint_names)}. "
                "Using the requested angles."
            )
            return None

        return [positions_by_name[name] for name in joint_names]

    @staticmethod
    def _nearest_equivalent_angle(
        target: float,
        reference: float,
        limits: Optional[tuple[float, float]] = None,
    ) -> float:
        if limits is not None:
            lower_limit, upper_limit = limits
            two_pi = 2.0 * math.pi
            min_turns = math.ceil((lower_limit - target) / two_pi)
            max_turns = math.floor((upper_limit - target) / two_pi)
            candidates = [
                target + turns * two_pi for turns in range(min_turns, max_turns + 1)
            ]
            if candidates:
                return min(candidates, key=lambda angle: abs(angle - reference))

        delta = target - reference
        return reference + math.atan2(math.sin(delta), math.cos(delta))

    def _nearest_equivalent_joint_positions(
        self, joint_positions: list[float], joint_names: list[str]
    ) -> list[float]:
        current_positions = self._current_joint_positions(joint_names)
        if current_positions is None:
            return list(joint_positions)

        return [
            self._nearest_equivalent_angle(
                target,
                current,
                self.JOINT_POSITION_LIMITS_RAD.get(name),
            )
            for name, target, current in zip(
                joint_names, joint_positions, current_positions, strict=True
            )
        ]

    @staticmethod
    def _format_joint_positions(
        joint_names: list[str], joint_positions_rad: list[float]
    ) -> str:
        return ", ".join(
            f"{name} {math.degrees(value):.1f} deg"
            for name, value in zip(joint_names, joint_positions_rad, strict=True)
        )

    def _joint_target_has_excessive_motion(
        self, label: str, joint_names: list[str], joint_positions_rad: list[float]
    ) -> bool:
        current_positions = self._current_joint_positions(joint_names)
        if current_positions is None:
            return False

        excessive = [
            (name, abs(target - current))
            for name, target, current in zip(
                joint_names, joint_positions_rad, current_positions, strict=True
            )
            if abs(target - current) > self.MAX_REASONABLE_JOINT_MOVE_RAD
        ]
        if not excessive:
            return False

        summary = ", ".join(
            f"{name} {math.degrees(delta):.1f} deg" for name, delta in excessive
        )
        self.get_logger().debug(
            f"Skipping {label}. It would move too far. {summary}."
        )
        return True

    def _pose_is_fresh(self, received_ns: Optional[int], label: str) -> bool:
        if received_ns is None:
            self.get_logger().warn(f"I have no timestamp for the {label} pose.")
            return False
        age_sec = (self.get_clock().now().nanoseconds - received_ns) / 1_000_000_000.0
        if age_sec <= self.MAX_POSE_AGE_SEC:
            return True
        self.get_logger().warn(
            f"The {label} pose is too old "
            f"at {age_sec:.2f} seconds old. Max is {self.MAX_POSE_AGE_SEC:.2f} seconds."
        )
        return False

    def _execute_joint_configuration(
        self,
        label: str,
        joint_positions_rad: list[float],
        joint_positions_deg: list[float],
    ) -> tuple[bool, str]:
        if not self._moveit_ready():
            return False, "MoveIt planning is not available."

        joint_names = ur.joint_names(prefix="")
        shortest_joint_positions_rad = self._nearest_equivalent_joint_positions(
            joint_positions_rad, joint_names
        )

        joints_summary = self._format_joint_positions(
            joint_names, shortest_joint_positions_rad
        )
        if any(
            abs(shortest - requested) > math.radians(1.0)
            for shortest, requested in zip(
                shortest_joint_positions_rad, joint_positions_rad, strict=True
            )
        ):
            requested_summary = ", ".join(
                f"{name} {value:.1f} deg"
                for name, value in zip(joint_names, joint_positions_deg, strict=True)
            )
            self.get_logger().info(
                f"{label} is using the nearest matching wrist angle. "
                f"Requested {requested_summary}. Using {joints_summary}."
            )

        self._publish_status(f"Moving to {label}. {joints_summary}.")

        try:
            with self._motion_active_guard():
                self._moveit.move_to_configuration(
                    joint_positions=shortest_joint_positions_rad,
                    joint_names=joint_names,
                    tolerance=self.JOINT_TOLERANCE,
                )
                success = self._wait_for_motion_completion(label)
        except Exception as exc:
            return False, f"MoveIt had a problem during {label}. {exc}"

        if success:
            self._publish_status(f"{label.capitalize()} done.")
            return True, f"{label.capitalize()} done."
        return False, f"{label.capitalize()} did not finish."

    def _send_gripper_command(self, label: str) -> tuple[bool, str]:
        if label == "open":
            width = 36.9
            force = 80
        elif label == "close":
            width = 0.0
            force = 31
        else:
            return False, f"I do not know the gripper command {label}."

        script = f"""sec codex_{label}():
  on_tool_xmlrpc = rpc_factory("xmlrpc", "http://localhost:41414")
  on_tool_xmlrpc.twofg_grip_external(0, {width}, {force}, 100)
end
"""

        try:
            with socket.create_connection(
                (self.robot_ip, self.ROBOT_SCRIPT_PORT),
                timeout=self.ROBOT_SCRIPT_TIMEOUT_SEC,
            ) as sock:
                sock.sendall(script.encode("utf-8"))
        except OSError as exc:
            return False, f"I could not send the gripper {label} command. {exc}"

        message = f"Gripper {label} command sent."
        self._publish_status(message)
        time.sleep(self.GRIPPER_SETTLE_SEC)
        return True, message

    def _joint_positions_from_state(
        self, joint_state, joint_names: list[str]
    ) -> Optional[list[float]]:
        positions_by_name = {
            name: float(position)
            for name, position in zip(joint_state.name, joint_state.position)
        }
        missing_joint_names = [
            name for name in joint_names if name not in positions_by_name
        ]
        if missing_joint_names:
            self.get_logger().warn(
                f"The IK result is missing {', '.join(missing_joint_names)}."
            )
            return None

        return [positions_by_name[name] for name in joint_names]

    def _ik_seed_candidates(self, joint_names: list[str]) -> list[Optional[list[float]]]:
        current_positions = self._current_joint_positions(joint_names)
        if current_positions is None:
            return [None]
        if not self.prefer_elbow_up_ik:
            return [current_positions]

        try:
            elbow_index = joint_names.index("elbow_joint")
        except ValueError:
            return [current_positions]

        limits = self.JOINT_POSITION_LIMITS_RAD.get("elbow_joint")
        elbow_seed_values = [
            self.elbow_up_seed_rad,
            -self.elbow_up_seed_rad,
            0.0,
            current_positions[elbow_index],
        ]
        seeds: list[Optional[list[float]]] = []
        seen: set[tuple[int, ...]] = set()
        for elbow_seed in elbow_seed_values:
            seed_positions = list(current_positions)
            seed_positions[elbow_index] = self._nearest_equivalent_angle(
                elbow_seed,
                current_positions[elbow_index],
                limits,
            )
            key = tuple(round(value * 1000.0) for value in seed_positions)
            if key in seen:
                continue
            seen.add(key)
            seeds.append(seed_positions)
        return seeds or [current_positions]

    def _wait_for_future(self, future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline:
            time.sleep(self.EXECUTION_POLL_INTERVAL_SEC)
        return future.done()

    def _compute_ik_joint_positions(
        self,
        pose: PoseStamped,
        start_joint_state,
        joint_names: list[str],
    ) -> Optional[list[float]]:
        future = self._moveit.compute_ik_async(
            position=pose.pose.position,
            quat_xyzw=pose.pose.orientation,
            ik_link_name=self.TARGET_LINK,
            start_joint_state=start_joint_state,
            wait_for_server_timeout_sec=self.MOVEIT_WAIT_SECONDS,
        )
        if future is None:
            return None
        if not self._wait_for_future(future, self.IK_WAIT_SECONDS):
            self.get_logger().warn(f"IK timed out after {self.IK_WAIT_SECONDS:.1f} seconds.")
            return None

        ik_joint_state = self._moveit.get_compute_ik_result(future)
        if ik_joint_state is None:
            return None

        ik_joint_positions = self._joint_positions_from_state(
            ik_joint_state, joint_names
        )
        if ik_joint_positions is None:
            return None

        shortest_joint_positions = self._nearest_equivalent_joint_positions(
            ik_joint_positions, joint_names
        )
        if self._joint_target_has_excessive_motion(
            "IK pose target", joint_names, shortest_joint_positions
        ):
            return None
        return shortest_joint_positions

    def _fk_link_z(
        self, joint_positions: list[float], link_name: str
    ) -> Optional[float]:
        future = self._moveit.compute_fk_async(
            joint_state=joint_positions,
            fk_link_names=[link_name],
        )
        if future is None:
            return None
        if not self._wait_for_future(future, self.IK_WAIT_SECONDS):
            self.get_logger().debug(
                f"FK timed out while checking {link_name} after {self.IK_WAIT_SECONDS:.1f} seconds."
            )
            return None

        poses = self._moveit.get_compute_fk_result(future, fk_link_names=[link_name])
        if poses is None:
            return None
        if not isinstance(poses, list):
            poses = [poses]
        if not poses:
            return None
        return float(poses[0].pose.position.z)

    def _make_pose_goal_constraints(self, pose: PoseStamped) -> Constraints:
        frame_id = pose.header.frame_id or self.BASE_LINK_NAME

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = frame_id
        position_constraint.link_name = self.TARGET_LINK
        position_constraint.weight = 1.0

        tolerance_region = SolidPrimitive()
        tolerance_region.type = SolidPrimitive.SPHERE
        tolerance_region.dimensions = [self.POSITION_TOLERANCE]
        position_constraint.constraint_region.primitives.append(tolerance_region)
        position_constraint.constraint_region.primitive_poses.append(pose.pose)

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = frame_id
        orientation_constraint.link_name = self.TARGET_LINK
        orientation_constraint.orientation = pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = self.ORIENTATION_TOLERANCE
        orientation_constraint.absolute_y_axis_tolerance = self.ORIENTATION_TOLERANCE
        orientation_constraint.absolute_z_axis_tolerance = self.ORIENTATION_TOLERANCE
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        return constraints

    def _plan_pose_final_joint_positions(
        self, pose: PoseStamped
    ) -> Optional[tuple[list[str], list[float]]]:
        if pose.header.frame_id and pose.header.frame_id != self.BASE_LINK_NAME:
            return None

        request = GetMotionPlan.Request()
        motion_request = request.motion_plan_request
        motion_request.group_name = self.GROUP_NAME
        motion_request.planner_id = self._moveit.planner_id
        motion_request.num_planning_attempts = self._moveit.num_planning_attempts
        motion_request.allowed_planning_time = self._moveit.allowed_planning_time
        motion_request.max_velocity_scaling_factor = self._moveit.max_velocity
        motion_request.max_acceleration_scaling_factor = self._moveit.max_acceleration
        if self._moveit.joint_state is not None:
            motion_request.start_state.joint_state = self._moveit.joint_state
        motion_request.goal_constraints.append(self._make_pose_goal_constraints(pose))

        future = self._plan_client.call_async(request)
        deadline = time.monotonic() + max(
            self.IK_WAIT_SECONDS,
            float(self._moveit.allowed_planning_time) + 1.0,
        )
        while not future.done() and time.monotonic() < deadline:
            time.sleep(self.EXECUTION_POLL_INTERVAL_SEC)

        if not future.done():
            self.get_logger().debug("The backup pose plan timed out.")
            return None

        response = future.result()
        if response is None:
            self.get_logger().debug("The backup pose plan gave no response.")
            return None

        motion_response = response.motion_plan_response
        if motion_response.error_code.val != self.MOVEIT_SUCCESS:
            self.get_logger().debug(
                "The backup pose plan failed with MoveIt error code "
                f"{motion_response.error_code.val}."
            )
            return None

        trajectory = motion_response.trajectory.joint_trajectory
        if not trajectory.points:
            self.get_logger().debug("The backup pose plan gave an empty path.")
            return None

        final_point = trajectory.points[-1]
        if len(trajectory.joint_names) != len(final_point.positions):
            self.get_logger().debug(
                "The backup pose plan gave joint names and positions that do not match."
            )
            return None

        return (
            list(trajectory.joint_names),
            [float(value) for value in final_point.positions],
        )

    def _move_to_pose_with_planned_configuration(self, pose: PoseStamped) -> bool:
        planned_target = self._plan_pose_final_joint_positions(pose)
        if planned_target is None:
            return False

        planned_joint_names, planned_joint_positions = planned_target
        positions_by_name = {
            name: position
            for name, position in zip(
                planned_joint_names, planned_joint_positions, strict=True
            )
        }
        joint_names = ur.joint_names(prefix="")
        missing_joint_names = [
            name for name in joint_names if name not in positions_by_name
        ]
        if missing_joint_names:
            self.get_logger().debug(
                "The backup pose plan is missing "
                f"{', '.join(missing_joint_names)}."
            )
            return False

        joint_positions = [positions_by_name[name] for name in joint_names]
        shortest_joint_positions = self._nearest_equivalent_joint_positions(
            joint_positions, joint_names
        )
        self.get_logger().debug(
            "Planned pose joint target is "
            f"{self._format_joint_positions(joint_names, shortest_joint_positions)}"
        )
        if self._joint_target_has_excessive_motion(
            "planned pose target", joint_names, shortest_joint_positions
        ):
            return False

        self._moveit.move_to_configuration(
            joint_positions=shortest_joint_positions,
            joint_names=joint_names,
            tolerance=self.JOINT_TOLERANCE,
        )
        return True

    def _move_to_pose_with_nearest_ik(self, pose: PoseStamped) -> bool:
        if pose.header.frame_id and pose.header.frame_id != self.BASE_LINK_NAME:
            return False

        joint_names = ur.joint_names(prefix="")
        ik_candidates: list[tuple[list[float], Optional[float]]] = []
        for seed in self._ik_seed_candidates(joint_names):
            joint_positions = self._compute_ik_joint_positions(
                pose,
                seed,
                joint_names,
            )
            if joint_positions is None:
                continue

            elbow_height = (
                self._fk_link_z(joint_positions, self.ELBOW_HEIGHT_LINK_NAME)
                if self.prefer_elbow_up_ik
                else None
            )
            ik_candidates.append((joint_positions, elbow_height))

        if not ik_candidates:
            return False

        shortest_joint_positions, elbow_height = max(
            ik_candidates,
            key=lambda candidate: (
                candidate[1] if candidate[1] is not None else float("-inf")
            ),
        )
        self.get_logger().debug(
            "IK joint target is "
            f"{self._format_joint_positions(joint_names, shortest_joint_positions)}"
        )
        if elbow_height is not None:
            self.get_logger().debug(
                f"Selected IK with {self.ELBOW_HEIGHT_LINK_NAME} at "
                f"{elbow_height:.3f} m from {len(ik_candidates)} candidates."
            )

        self._moveit.move_to_configuration(
            joint_positions=shortest_joint_positions,
            joint_names=joint_names,
            tolerance=self.JOINT_TOLERANCE,
        )
        return True

    def _execute_pose(
        self,
        label: str,
        pose: Optional[PoseStamped],
        received_ns: Optional[int],
        require_fresh_pose: bool = True,
        use_nearest_ik: bool = True,
        planner_label: str = "",
    ) -> tuple[bool, str]:
        if pose is None:
            return False, f"No cached {label} pose yet."
        if not self._moveit_ready():
            return False, "MoveIt planning is not available."
        if require_fresh_pose and not self._pose_is_fresh(received_ns, label):
            return (
                False,
                f"{label.capitalize()} pose is too old. Reacquire the target first.",
            )

        candidates = (
            self._build_approach_candidates(pose)
            if label in {"approach", "camera-center approach"}
            else [(0.0, 0.0, 0.0, pose)]
        )

        last_failure_message = f"I could not move to {label}."

        try:
            with self._motion_active_guard():
                for dx, dy, dz, candidate_pose in candidates:
                    candidate_label = self._format_pose_candidate_label(dx, dy, dz)
                    planner_text = f" with {planner_label}" if planner_label else ""
                    if candidate_label:
                        self._publish_status(
                            f"Trying a nearby {label}{candidate_label}{planner_text}. "
                            f"{candidate_pose.pose.position.x:.3f}, "
                            f"{candidate_pose.pose.position.y:.3f}, "
                            f"{candidate_pose.pose.position.z:.3f}."
                        )
                    else:
                        self._publish_status(
                            f"Moving to {label}{planner_text}. "
                            f"{candidate_pose.pose.position.x:.3f}, "
                            f"{candidate_pose.pose.position.y:.3f}, "
                            f"{candidate_pose.pose.position.z:.3f}. "
                            f"Frame is {candidate_pose.header.frame_id or self.BASE_LINK_NAME}."
                        )
                    used_nearest_ik = False
                    if use_nearest_ik:
                        used_nearest_ik = self._move_to_pose_with_nearest_ik(
                            candidate_pose
                        )
                        if (
                            not used_nearest_ik
                            and dx == 0.0
                            and dy == 0.0
                            and dz == 0.0
                        ):
                            self.get_logger().debug(
                                "Quick IK did not work for the main pose. "
                                "Trying a planned joint move."
                            )
                            used_nearest_ik = (
                                self._move_to_pose_with_planned_configuration(
                                    candidate_pose
                                )
                            )
                    if use_nearest_ik and not used_nearest_ik:
                        self.get_logger().debug(
                            f"No reasonable IK joint target for {label}"
                            f"{candidate_label}. Trying a pose goal."
                        )
                    if not used_nearest_ik:
                        self._moveit.move_to_pose(
                            pose=candidate_pose,
                            target_link=self.TARGET_LINK,
                            tolerance_position=self.POSITION_TOLERANCE,
                            tolerance_orientation=self.ORIENTATION_TOLERANCE,
                        )
                    success = self._wait_for_motion_completion(label)
                    if not success and used_nearest_ik:
                        self.get_logger().debug(
                            f"{label.capitalize()} IK joint move failed. "
                            "Trying a pose goal."
                        )
                        self._moveit.move_to_pose(
                            pose=candidate_pose,
                            target_link=self.TARGET_LINK,
                            tolerance_position=self.POSITION_TOLERANCE,
                            tolerance_orientation=self.ORIENTATION_TOLERANCE,
                        )
                        success = self._wait_for_motion_completion(label)
                    if success:
                        if dx == 0.0 and dy == 0.0 and dz == 0.0:
                            self._publish_status(f"{label.capitalize()} done.")
                            return True, f"{label.capitalize()} done."
                        self._publish_status(
                            f"{label.capitalize()} worked with nearby pose "
                            f"{dx:.3f}, {dy:.3f}, {dz:.3f}."
                        )
                        return (
                            True,
                            f"{label.capitalize()} worked with nearby pose.",
                        )

                    last_failure_message = f"{label.capitalize()} did not finish{candidate_label}."
        except Exception as exc:
            return False, f"MoveIt had a problem during {label}. {exc}"

        return False, last_failure_message

    def _execute_linear_pose(self, label: str, pose: PoseStamped) -> tuple[bool, str]:
        with self._moveit_planner_guard(
            self.LINEAR_GRASP_PIPELINE_ID,
            self.LINEAR_GRASP_PLANNER_ID,
        ):
            return self._execute_pose(
                label,
                pose,
                None,
                require_fresh_pose=False,
                use_nearest_ik=False,
                planner_label="Pilz LIN",
            )

    @staticmethod
    def _clone_pose(pose: PoseStamped) -> PoseStamped:
        pose_copy = PoseStamped()
        pose_copy.header.stamp = pose.header.stamp
        pose_copy.header.frame_id = pose.header.frame_id
        pose_copy.pose.position.x = float(pose.pose.position.x)
        pose_copy.pose.position.y = float(pose.pose.position.y)
        pose_copy.pose.position.z = float(pose.pose.position.z)
        pose_copy.pose.orientation.x = float(pose.pose.orientation.x)
        pose_copy.pose.orientation.y = float(pose.pose.orientation.y)
        pose_copy.pose.orientation.z = float(pose.pose.orientation.z)
        pose_copy.pose.orientation.w = float(pose.pose.orientation.w)
        return pose_copy

    @staticmethod
    def _quaternion_from_rotation_vector(
        rx: float, ry: float, rz: float
    ) -> tuple[float, float, float, float]:
        angle = math.sqrt(rx * rx + ry * ry + rz * rz)
        if angle <= 1e-9:
            return 0.0, 0.0, 0.0, 1.0

        scale = math.sin(0.5 * angle) / angle
        return rx * scale, ry * scale, rz * scale, math.cos(0.5 * angle)

    def _selected_sort_class(self) -> tuple[Optional[str], str]:
        class_name = self._normalize_class_name(self.latest_target_class)
        if not class_name:
            return None, "No selected object class yet."
        if class_name not in self.sort_drop_classes:
            return (
                None,
                f"The selected object class {class_name} has no sorting drop slot.",
            )
        return class_name, ""

    def _build_sort_drop_pose(self, class_name: str) -> tuple[PoseStamped, int]:
        class_index = self.sort_drop_classes.index(class_name)
        used_count = self.sort_drop_counts.get(class_name, 0)
        slot_in_class = min(used_count, self.SORT_DROP_SLOTS_PER_CLASS - 1)
        slot_index = class_index * self.SORT_DROP_SLOTS_PER_CLASS + slot_in_class

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.sort_drop_frame
        pose.pose.position.x = self.sort_drop_base_xyz_m[0]
        pose.pose.position.y = (
            self.sort_drop_base_xyz_m[1] - slot_index * self.sort_drop_slot_offset_y
        )
        pose.pose.position.z = self._sort_drop_release_z()

        qx, qy, qz, qw = self._quaternion_from_rotation_vector(
            self.sort_drop_base_rotvec_rad[0],
            self.sort_drop_base_rotvec_rad[1],
            self.sort_drop_base_rotvec_rad[2],
        )
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose, slot_in_class + 1

    def _sort_drop_release_z(self) -> float:
        return self.sort_drop_table_top_z + self.sort_drop_release_clearance_z

    def _mark_sort_drop_used(self, class_name: str) -> None:
        self.sort_drop_counts[class_name] = self.sort_drop_counts.get(class_name, 0) + 1

    def _build_sort_drop_above_pose(self, drop_pose: PoseStamped) -> PoseStamped:
        above_pose = self._clone_pose(drop_pose)
        above_pose.pose.position.z = (
            float(drop_pose.pose.position.z) + self.sort_drop_approach_clearance_z
        )
        return above_pose

    def _execute_sort_drop_pose(self) -> tuple[bool, str, Optional[str]]:
        class_name, error_message = self._selected_sort_class()
        if class_name is None:
            return False, error_message, None

        self.last_sort_drop_lift_pose = None
        drop_pose, slot_number = self._build_sort_drop_pose(class_name)
        above_pose = self._build_sort_drop_above_pose(drop_pose)
        used_count = self.sort_drop_counts.get(class_name, 0)
        if used_count >= self.SORT_DROP_SLOTS_PER_CLASS:
            self.get_logger().warn(
                f"All {class_name} sorting slots are already used. Reusing slot "
                f"{self.SORT_DROP_SLOTS_PER_CLASS}."
            )

        ok, message = self._execute_pose(
            f"above sort drop {class_name} slot {slot_number}",
            above_pose,
            None,
            require_fresh_pose=False,
        )
        if not ok:
            return False, message, class_name

        with self._moveit_speed_guard(
            self.grasp_velocity_scaling, self.grasp_acceleration_scaling
        ):
            ok, message = self._execute_linear_pose(
                f"lower to sort drop {class_name} slot {slot_number}",
                drop_pose,
            )
        if not ok:
            return False, message, class_name
        self.last_sort_drop_lift_pose = self._clone_pose(above_pose)
        return True, message, class_name

    @staticmethod
    def _flip_pose_yaw(pose: PoseStamped) -> None:
        qx = float(pose.pose.orientation.x)
        qy = float(pose.pose.orientation.y)
        qz = float(pose.pose.orientation.z)
        qw = float(pose.pose.orientation.w)
        pose.pose.orientation.x = -qy
        pose.pose.orientation.y = qx
        pose.pose.orientation.z = qw
        pose.pose.orientation.w = -qz

    @staticmethod
    def _normalized_quaternion_from_pose(
        pose: PoseStamped,
    ) -> tuple[float, float, float, float]:
        qx = float(pose.pose.orientation.x)
        qy = float(pose.pose.orientation.y)
        qz = float(pose.pose.orientation.z)
        qw = float(pose.pose.orientation.w)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 0.0:
            return 0.0, 0.0, 0.0, 1.0
        return qx / norm, qy / norm, qz / norm, qw / norm

    @classmethod
    def _quaternion_dot(cls, a: PoseStamped, b: PoseStamped) -> float:
        ax, ay, az, aw = cls._normalized_quaternion_from_pose(a)
        bx, by, bz, bw = cls._normalized_quaternion_from_pose(b)
        return ax * bx + ay * by + az * bz + aw * bw

    @classmethod
    def _orientation_distance(cls, a: PoseStamped, b: PoseStamped) -> float:
        dot = abs(cls._quaternion_dot(a, b))
        dot = min(max(dot, -1.0), 1.0)
        return 2.0 * math.acos(dot)

    @classmethod
    def _align_quaternion_hemisphere(
        cls, reference_pose: PoseStamped, pose: PoseStamped
    ) -> None:
        if cls._quaternion_dot(reference_pose, pose) >= 0.0:
            return
        pose.pose.orientation.x *= -1.0
        pose.pose.orientation.y *= -1.0
        pose.pose.orientation.z *= -1.0
        pose.pose.orientation.w *= -1.0

    def _nearest_half_turn_grasp_pose(
        self, reference_pose: PoseStamped, grasp_pose: PoseStamped
    ) -> PoseStamped:
        original_pose = self._clone_pose(grasp_pose)
        flipped_pose = self._clone_pose(grasp_pose)
        self._flip_pose_yaw(flipped_pose)

        original_distance = self._orientation_distance(reference_pose, original_pose)
        flipped_distance = self._orientation_distance(reference_pose, flipped_pose)
        if flipped_distance + 1e-6 < original_distance:
            self.get_logger().debug(
                "Using the half turn grasp yaw to reduce wrist rotation. "
                f"{math.degrees(original_distance):.1f} deg to "
                f"{math.degrees(flipped_distance):.1f} deg."
            )
            self._align_quaternion_hemisphere(reference_pose, flipped_pose)
            return flipped_pose

        self._align_quaternion_hemisphere(reference_pose, original_pose)
        return original_pose

    @staticmethod
    def _format_pose_candidate_label(dx: float, dy: float, dz: float) -> str:
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            return ""
        return f" with offset {dx:.3f}, {dy:.3f}, {dz:.3f}"

    def _build_approach_candidates(
        self, pose: PoseStamped
    ) -> list[tuple[float, float, float, PoseStamped]]:
        offsets: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
        if not self.approach_fallback_enabled:
            return [(0.0, 0.0, 0.0, self._clone_pose(pose))]

        seen_offsets = {(0.0, 0.0, 0.0)}
        for z_level in range(self.approach_fallback_z_levels + 1):
            dz = round(z_level * self.approach_fallback_z_step, 6)
            if dz > 0.0 and (0.0, 0.0, dz) not in seen_offsets:
                offsets.append((0.0, 0.0, dz))
                seen_offsets.add((0.0, 0.0, dz))

            for xy_level in range(1, self.approach_fallback_xy_levels + 1):
                step = round(xy_level * self.approach_fallback_xy_step, 6)
                ring_offsets = [
                    (step, 0.0, dz),
                    (-step, 0.0, dz),
                    (0.0, step, dz),
                    (0.0, -step, dz),
                    (step, step, dz),
                    (step, -step, dz),
                    (-step, step, dz),
                    (-step, -step, dz),
                ]
                for offset in ring_offsets:
                    if offset in seen_offsets:
                        continue
                    offsets.append(offset)
                    seen_offsets.add(offset)

        candidates: list[tuple[float, float, float, PoseStamped]] = []
        for dx, dy, dz in offsets:
            candidate_pose = self._clone_pose(pose)
            candidate_pose.pose.position.x += dx
            candidate_pose.pose.position.y += dy
            candidate_pose.pose.position.z += dz
            candidates.append((dx, dy, dz, candidate_pose))

        return candidates

    def _get_fresh_pick_pose_snapshot(
        self,
    ) -> tuple[Optional[PoseStamped], Optional[PoseStamped], str]:
        if self.latest_grasp_pose is None:
            return None, None, "No cached grasp pose yet."
        if self.latest_approach_pose is None:
            return None, None, "No cached approach pose yet."
        if not self._pose_is_fresh(self.latest_grasp_received_ns, "grasp"):
            return (
                None,
                None,
                "Grasp pose is too old. Reacquire the target first.",
            )
        if not self._pose_is_fresh(self.latest_approach_received_ns, "approach"):
            return (
                None,
                None,
                "Approach pose is too old. Reacquire the target first.",
            )

        return (
            self._clone_pose(self.latest_approach_pose),
            self._clone_pose(self.latest_grasp_pose),
            "",
        )

    def _wait_for_reacquired_pick_pose_snapshot(
        self, min_received_ns: int, reason: str
    ) -> tuple[Optional[PoseStamped], Optional[PoseStamped], str]:
        wait_sec = self.approach_to_grasp_wait_sec
        if wait_sec > 0.0:
            self._publish_status(f"Waiting {wait_sec:.2f} seconds {reason}.")

        latest_snapshot: tuple[Optional[PoseStamped], Optional[PoseStamped], str] = (
            None,
            None,
            "",
        )
        deadline = time.monotonic() + wait_sec
        while True:
            if (
                self.latest_approach_received_ns is not None
                and self.latest_grasp_received_ns is not None
                and self.latest_approach_received_ns > min_received_ns
                and self.latest_grasp_received_ns > min_received_ns
            ):
                latest_snapshot = self._get_fresh_pick_pose_snapshot()
                if wait_sec <= 0.0:
                    return latest_snapshot

            if wait_sec <= 0.0 or time.monotonic() >= deadline:
                break

            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0.0:
                break
            time.sleep(min(self.EXECUTION_POLL_INTERVAL_SEC, remaining_sec))

        if latest_snapshot[0] is not None and latest_snapshot[1] is not None:
            return latest_snapshot

        return None, None, f"No updated pick pose was received {reason}."

    def _wait_for_search_target(
        self, min_received_ns: int, label: str
    ) -> tuple[bool, str]:
        wait_sec = self.search_pose_wait_sec
        if wait_sec > 0.0:
            self._publish_status(
                f"Searching workspace. Waiting {wait_sec:.2f} seconds at {label}."
            )

        saw_2d_detection = False
        deadline = time.monotonic() + wait_sec
        while True:
            if (
                self.latest_approach_received_ns is not None
                and self.latest_grasp_received_ns is not None
                and self.latest_target_class_received_ns is not None
                and self.latest_approach_received_ns > min_received_ns
                and self.latest_grasp_received_ns > min_received_ns
                and self.latest_target_class_received_ns > min_received_ns
            ):
                approach_pose, grasp_pose, error_message = (
                    self._get_fresh_pick_pose_snapshot()
                )
                if approach_pose is not None and grasp_pose is not None:
                    return True, f"Target found from {label}."
                return False, error_message

            if (
                self.latest_yolo_detection_received_ns is not None
                and self.latest_yolo_detection_received_ns > min_received_ns
                and self.latest_yolo_detection_count > 0
            ):
                saw_2d_detection = True

            if wait_sec <= 0.0 or time.monotonic() >= deadline:
                break

            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0.0:
                break
            time.sleep(min(self.EXECUTION_POLL_INTERVAL_SEC, remaining_sec))

        if saw_2d_detection:
            return (
                False,
                f"Found a 2D detection at {label}. No 3D pick pose yet.",
            )
        return False, f"No target found from {label}."

    def _search_workspace(self) -> tuple[bool, str]:
        search_targets = [
            (
                "search start pose",
                self.search_start_joint_positions_rad,
                self.search_start_joint_positions_deg,
            )
        ]
        for index, offset_deg in enumerate(self.search_look_offsets_deg, start=1):
            joint_positions_deg = [
                start_value + offset_value
                for start_value, offset_value in zip(
                    self.search_start_joint_positions_deg,
                    offset_deg,
                    strict=True,
                )
            ]
            search_targets.append(
                (
                    f"search look {index}",
                    [math.radians(value) for value in joint_positions_deg],
                    joint_positions_deg,
                )
            )

        for index, (joint_positions_rad, joint_positions_deg) in enumerate(
            zip(
                self.search_joint_positions_rad,
                self.search_joint_positions_deg,
                strict=True,
            ),
            start=1,
        ):
            search_targets.append(
                (
                    f"search pose {index}",
                    joint_positions_rad,
                    joint_positions_deg,
                )
            )

        last_message = "No target found during workspace search."
        for label, joint_positions_rad, joint_positions_deg in search_targets:
            before_move_ns = self.get_clock().now().nanoseconds
            ok, message = self._execute_joint_configuration(
                label,
                joint_positions_rad,
                joint_positions_deg,
            )
            if not ok:
                return False, f"Workspace search could not move to {label}. {message}"

            ok, message = self._wait_for_search_target(before_move_ns, label)
            if ok:
                self._publish_status(message)
                return True, message
            last_message = message

        self._publish_status(last_message)
        return False, last_message

    def _pre_grasp_z(
        self, approach_pose: PoseStamped, grasp_pose: PoseStamped
    ) -> float:
        approach_z = float(approach_pose.pose.position.z)
        grasp_z = float(grasp_pose.pose.position.z)
        return max(grasp_z, min(approach_z, grasp_z + self.pre_grasp_clearance_z))

    def _build_camera_offset_pre_grasp_pose(
        self, approach_pose: PoseStamped, grasp_pose: PoseStamped
    ) -> PoseStamped:
        pre_grasp_pose = self._clone_pose(approach_pose)
        pre_grasp_pose.pose.position.z = self._pre_grasp_z(approach_pose, grasp_pose)
        return pre_grasp_pose

    def _build_grasp_above_pose(
        self, pre_grasp_pose: PoseStamped, grasp_pose: PoseStamped
    ) -> PoseStamped:
        grasp_above_pose = self._clone_pose(pre_grasp_pose)
        grasp_above_pose.pose.position.x = float(grasp_pose.pose.position.x)
        grasp_above_pose.pose.position.y = float(grasp_pose.pose.position.y)
        return grasp_above_pose

    def _build_grasp_above_candidates(
        self,
        pre_grasp_pose: PoseStamped,
        grasp_pose: PoseStamped,
        approach_pose: PoseStamped,
    ) -> list[PoseStamped]:
        base_z = float(pre_grasp_pose.pose.position.z)
        grasp_z = float(grasp_pose.pose.position.z)
        approach_z = float(approach_pose.pose.position.z)
        min_clearance_z = min(self.pre_grasp_clearance_z, 0.02)
        min_z = grasp_z + min_clearance_z
        max_z = max(base_z, approach_z)
        z_offsets = [0.0, 0.02, 0.04, -0.01, -0.02, 0.06]

        candidates: list[PoseStamped] = []
        seen_z: set[float] = set()
        for z_offset in z_offsets:
            z = round(min(max(base_z + z_offset, min_z), max_z), 6)
            if z in seen_z:
                continue
            seen_z.add(z)
            candidate = self._build_grasp_above_pose(pre_grasp_pose, grasp_pose)
            candidate.pose.position.z = z
            candidates.append(candidate)
        return candidates

    def _execute_grasp_above_candidates(
        self, candidates: list[PoseStamped]
    ) -> tuple[bool, str, Optional[PoseStamped]]:
        last_message = "Could not move above the object."
        for index, candidate in enumerate(candidates):
            label = (
                "grasp above object"
                if index == 0
                else f"grasp above object, try {index + 1}"
            )
            ok, message = self._execute_linear_pose(label, candidate)
            if ok:
                return True, message, candidate
            last_message = message
        return False, last_message, None

    def _build_grasp_rotate_pose(
        self, grasp_above_pose: PoseStamped, grasp_pose: PoseStamped
    ) -> PoseStamped:
        grasp_rotate_pose = self._clone_pose(grasp_above_pose)
        grasp_rotate_pose.pose.orientation = grasp_pose.pose.orientation
        return grasp_rotate_pose

    def _build_post_grasp_lift_pose(
        self, approach_pose: PoseStamped, grasp_pose: PoseStamped
    ) -> PoseStamped:
        lift_pose = self._clone_pose(grasp_pose)
        lift_pose.pose.position.z = self._pre_grasp_z(approach_pose, grasp_pose)
        return lift_pose

    def _execute_grasp(
        self,
        approach_pose: Optional[PoseStamped] = None,
        grasp_pose: Optional[PoseStamped] = None,
    ) -> tuple[bool, str]:
        if approach_pose is None or grasp_pose is None:
            approach_pose, grasp_pose, error_message = (
                self._get_fresh_pick_pose_snapshot()
            )
            if approach_pose is None or grasp_pose is None:
                return False, error_message

        approach_snapshot = self._clone_pose(approach_pose)
        grasp_snapshot = self._clone_pose(grasp_pose)

        with self._moveit_speed_guard(
            self.grasp_velocity_scaling, self.grasp_acceleration_scaling
        ):
            pre_grasp_snapshot = self._build_camera_offset_pre_grasp_pose(
                approach_snapshot, grasp_snapshot
            )
            grasp_above_candidates = self._build_grasp_above_candidates(
                pre_grasp_snapshot, grasp_snapshot, approach_snapshot
            )

            ok, message, grasp_above_pose = self._execute_grasp_above_candidates(
                grasp_above_candidates
            )
            if not ok or grasp_above_pose is None:
                return False, message

            grasp_snapshot = self._nearest_half_turn_grasp_pose(
                grasp_above_pose, grasp_snapshot
            )
            grasp_rotate_pose = self._build_grasp_rotate_pose(
                grasp_above_pose, grasp_snapshot
            )

            ok, message = self._execute_pose(
                "rotate above object",
                grasp_rotate_pose,
                None,
                require_fresh_pose=False,
            )
            if not ok:
                return False, message

            ok, message = self._execute_linear_pose("grasp", grasp_snapshot)
            if ok:
                self.last_completed_grasp_pose = self._clone_pose(grasp_snapshot)
            return ok, message

    def _execute_centered_approach(self) -> tuple[bool, str]:
        approach_pose, grasp_pose, error_message = self._get_fresh_pick_pose_snapshot()
        if approach_pose is None or grasp_pose is None:
            return False, error_message

        ok, message = self._execute_pose(
            "approach",
            approach_pose,
            None,
            require_fresh_pose=False,
        )
        if not ok:
            return False, message

        approach_completed_ns = self.get_clock().now().nanoseconds
        approach_pose, grasp_pose, error_message = (
            self._wait_for_reacquired_pick_pose_snapshot(
                approach_completed_ns,
                "to recenter camera over target",
            )
        )
        if approach_pose is None or grasp_pose is None:
            return False, error_message

        return self._execute_pose(
            "camera-center approach",
            approach_pose,
            None,
            require_fresh_pose=False,
        )

    def _execute_pick_pipeline(self) -> tuple[bool, str]:
        ok, message = self._send_gripper_command("open")
        if not ok:
            return False, message

        ok, message = self._execute_centered_approach()
        if not ok:
            return False, message

        recenter_completed_ns = self.get_clock().now().nanoseconds
        approach_pose, grasp_pose, error_message = (
            self._wait_for_reacquired_pick_pose_snapshot(
                recenter_completed_ns,
                "before grasp",
            )
        )
        if approach_pose is None or grasp_pose is None:
            return False, error_message

        self.last_completed_grasp_pose = None
        ok, message = self._execute_grasp(approach_pose, grasp_pose)
        if not ok:
            return False, message

        ok, message = self._send_gripper_command("close")
        if not ok:
            return False, message

        final_grasp_pose = self.last_completed_grasp_pose or grasp_pose
        lift_pose = self._build_post_grasp_lift_pose(approach_pose, final_grasp_pose)
        with self._moveit_speed_guard(
            self.grasp_velocity_scaling, self.grasp_acceleration_scaling
        ):
            return self._execute_linear_pose("lift after grasp", lift_pose)

    def _run_dice_test(self) -> tuple[bool, str]:
        while rclpy.ok() and not self._dice_test_stop_requested:
            self._publish_status("Dice test is searching the workspace.")

            ok, message = self._search_workspace()
            if not ok:
                return (
                    False,
                    f"Dice test stopped during workspace search. {message}",
                )

            self._publish_status("Dice test is picking.")

            ok, message = self._execute_pick_pipeline()
            if not ok:
                return (
                    False,
                    f"Dice test stopped during pick. {message}",
                )

            ok, message = self._execute_joint_configuration(
                "dice drop pose",
                self.dice_drop_joint_positions_rad,
                self.dice_drop_joint_positions_deg,
            )
            if not ok:
                return (
                    False,
                    f"Dice test stopped while moving to the drop pose. {message}",
                )

            ok, message = self._send_gripper_command("open")
            if not ok:
                return (
                    False,
                    f"Dice test stopped while opening the gripper. {message}",
                )

            if self.dice_repick_wait_sec > 0.0:
                self._publish_status(
                    f"Waiting {self.dice_repick_wait_sec:.2f} seconds before the next dice pick."
                )
                time.sleep(self.dice_repick_wait_sec)

        if self._dice_test_stop_requested:
            return True, "Dice test stopped by request."
        return True, "Dice test stopped because ROS is shutting down."

    def _run_sorting(self) -> tuple[bool, str]:
        self.sort_drop_counts = {class_name: 0 for class_name in self.sort_drop_classes}

        while rclpy.ok() and not self._sorting_stop_requested:
            self._publish_status("Sorting is searching the workspace.")

            ok, message = self._search_workspace()
            if not ok:
                return (
                    False,
                    f"Sorting stopped during workspace search. {message}",
                )

            self._publish_status("Sorting is picking.")

            ok, message = self._execute_pick_pipeline()
            if not ok:
                return (
                    False,
                    f"Sorting stopped during pick. {message}",
                )

            ok, message, dropped_class_name = self._execute_sort_drop_pose()
            if not ok:
                return (
                    False,
                    f"Sorting stopped while moving to the drop pose. {message}",
                )

            ok, message = self._send_gripper_command("open")
            if not ok:
                return (
                    False,
                    f"Sorting stopped while opening the gripper. {message}",
                )

            if self.last_sort_drop_lift_pose is not None:
                with self._moveit_speed_guard(
                    self.grasp_velocity_scaling, self.grasp_acceleration_scaling
                ):
                    ok, message = self._execute_linear_pose(
                        "lift after sort drop",
                        self.last_sort_drop_lift_pose,
                    )
                if not ok:
                    return (
                        False,
                        f"Sorting stopped while lifting after the drop. {message}",
                    )

            if dropped_class_name is not None:
                self._mark_sort_drop_used(dropped_class_name)

            if self.dice_repick_wait_sec > 0.0:
                self._publish_status(
                    f"Waiting {self.dice_repick_wait_sec:.2f} seconds before the next sorted pick."
                )
                time.sleep(self.dice_repick_wait_sec)

        if self._sorting_stop_requested:
            return True, "Sorting stopped by request."
        return True, "Sorting stopped because ROS is shutting down."

    def _handle_execute_approach(self, request, response):
        del request
        response.success, response.message = self._execute_pose(
            "approach",
            self.latest_approach_pose,
            self.latest_approach_received_ns,
        )
        return response

    def _handle_execute_centered_approach(self, request, response):
        del request
        response.success, response.message = self._execute_centered_approach()
        return response

    def _handle_execute_grasp(self, request, response):
        del request
        response.success, response.message = self._execute_grasp()
        return response

    def _handle_execute_pick(self, request, response):
        del request
        response.success, response.message = self._execute_pick_pipeline()
        return response

    def _handle_search_workspace(self, request, response):
        del request
        response.success, response.message = self._search_workspace()
        return response

    def _handle_run_dice_test(self, request, response):
        del request
        if self._dice_test_running:
            response.success = False
            response.message = "Dice test is already running."
            return response
        if self._sorting_running:
            response.success = False
            response.message = "Sorting is already running."
            return response

        self._dice_test_running = True
        self._dice_test_stop_requested = False
        try:
            response.success, response.message = self._run_dice_test()
        finally:
            self._dice_test_running = False
            self._dice_test_stop_requested = False
        return response

    def _handle_stop_dice_test(self, request, response):
        del request
        if not self._dice_test_running:
            response.success = True
            response.message = "Dice test is not running."
            return response

        self._dice_test_stop_requested = True
        response.success = True
        response.message = "Stopping dice test. Waiting for the current step to finish."
        return response

    def _handle_run_sorting(self, request, response):
        del request
        if self._sorting_running:
            response.success = False
            response.message = "Sorting is already running."
            return response
        if self._dice_test_running:
            response.success = False
            response.message = "Dice test is already running."
            return response

        self._sorting_running = True
        self._sorting_stop_requested = False
        try:
            response.success, response.message = self._run_sorting()
        finally:
            self._sorting_running = False
            self._sorting_stop_requested = False
        return response

    def _handle_stop_sorting(self, request, response):
        del request
        if not self._sorting_running:
            response.success = True
            response.message = "Sorting is not running."
            return response

        self._sorting_stop_requested = True
        response.success = True
        response.message = "Stopping sorting. Waiting for the current step to finish."
        return response

    def _handle_move_to_start_pose(self, request, response):
        del request
        response.success, response.message = self._execute_joint_configuration(
            "start pose",
            self.ready_joint_positions_rad,
            self.ready_joint_positions_deg,
        )
        return response

    def _handle_open_gripper(self, request, response):
        del request
        response.success, response.message = self._send_gripper_command("open")
        return response

    def _handle_close_gripper(self, request, response):
        del request
        response.success, response.message = self._send_gripper_command("close")
        return response


def main():
    rclpy.init()
    node = PickMoveItExecutorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
