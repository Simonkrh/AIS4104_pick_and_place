#!/usr/bin/env python3
from __future__ import annotations

import ast
import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetMotionPlan
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection2DArray

try:
    from .pick_executor.gripper_controller import GripperControllerMixin
    from .pick_executor.motion_executor import MotionExecutorMixin
    from .pick_executor.pick_pipeline import PickPipelineMixin
    from .pick_executor.pose_utils import PoseUtilsMixin
    from .pick_executor.search_controller import SearchControllerMixin
    from .pick_executor.sorting_pipeline import SortingPipelineMixin
    from .pick_executor.dice_test_executor import DiceTestMixin
except ImportError:
    from pick_executor.gripper_controller import GripperControllerMixin
    from pick_executor.motion_executor import MotionExecutorMixin
    from pick_executor.pick_pipeline import PickPipelineMixin
    from pick_executor.pose_utils import PoseUtilsMixin
    from pick_executor.search_controller import SearchControllerMixin
    from pick_executor.sorting_pipeline import SortingPipelineMixin
    from pick_executor.dice_test_executor import DiceTestMixin


class PickMoveItExecutorNode(
    PoseUtilsMixin,
    MotionExecutorMixin,
    GripperControllerMixin,
    SearchControllerMixin,
    PickPipelineMixin,
    SortingPipelineMixin,
    DiceTestMixin,
    Node,
):
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
        self.prefer_elbow_up_ik = bool(self.get_parameter("prefer_elbow_up_ik").value)
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
        self.get_logger().info(
            f"Reading target classes from {self.target_class_topic}."
        )
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
            f"Dice re pick wait is {self.dice_repick_wait_sec:.2f} s."
        )
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
        self.get_logger().info(
            f"Sending gripper scripts to {self.robot_ip} on port 30002."
        )

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
