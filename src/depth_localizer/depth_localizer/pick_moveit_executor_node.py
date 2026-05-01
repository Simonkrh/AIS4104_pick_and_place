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


class PickMoveItExecutorNode(Node):
    GROUP_NAME = "ur_manipulator"
    BASE_LINK_NAME = "base_link"
    END_EFFECTOR_NAME = "gripper_tcp"
    TARGET_LINK = "gripper_tcp"
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

    def __init__(self):
        super().__init__("pick_moveit_executor_node")

        self.declare_parameter("approach_topic", "/pick_approach_pose")
        self.declare_parameter("grasp_topic", "/pick_grasp_pose")
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
            "[[0.0, 0.0, 0.0, 5.0, 10.0, 0.0], "
            "[0.0, 0.0, 0.0, 5.0, -8.0, 0.0], "
            "[0.0, 0.0, 0.0, 10.0, 0.0, 0.0], "
            "[0.0, -6.0, 0.0, -15.0, -4.0, 6.0], "
            "[0.0, 0.0, 0.0, -10.0, 14.0, -6.0]]",
        )
        self.declare_parameter("search_joint_positions_deg", "[]")
        self.declare_parameter("search_pose_wait_sec", 1.0)
        self.declare_parameter("robot_ip", self.DEFAULT_ROBOT_IP)

        self.approach_topic = str(self.get_parameter("approach_topic").value)
        self.grasp_topic = str(self.get_parameter("grasp_topic").value)
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

        self.callback_group = ReentrantCallbackGroup()
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.motion_active_pub = self.create_publisher(
            Bool, self.motion_active_topic, 10
        )

        self.latest_approach_pose: Optional[PoseStamped] = None
        self.latest_grasp_pose: Optional[PoseStamped] = None
        self.latest_approach_received_ns: Optional[int] = None
        self.latest_grasp_received_ns: Optional[int] = None
        self._motion_active_depth = 0
        self._dice_test_running = False
        self._dice_test_stop_requested = False

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
            "~/move_to_ready_pose",
            self._handle_move_to_ready_pose,
            callback_group=self.callback_group,
        )
        self.create_service(
            Trigger,
            "~/move_to_start_pose",
            self._handle_move_to_ready_pose,
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

        self.get_logger().info(f"Subscribing approach pose: {self.approach_topic}")
        self.get_logger().info(f"Subscribing grasp pose: {self.grasp_topic}")
        self.get_logger().info(f"Publishing execution status: {self.status_topic}")
        self.get_logger().info(
            f"Publishing motion active flag: {self.motion_active_topic}"
        )
        self.get_logger().info(
            "Services: ~/execute_approach, ~/execute_centered_approach, "
            "~/execute_grasp, ~/execute_pick, ~/search_workspace, "
            "~/run_dice_test, ~/stop_dice_test, ~/move_to_ready_pose, "
            "~/move_to_start_pose, ~/open_gripper, ~/close_gripper"
        )
        self.get_logger().info(
            f"Using MoveIt group {self.GROUP_NAME} from {self.BASE_LINK_NAME} "
            f"to {self.TARGET_LINK}."
        )
        self.get_logger().info(
            f"Ready pose joint targets (deg): {self.ready_joint_positions_deg}"
        )
        self.get_logger().info(
            f"Dice drop joint targets (deg): {self.dice_drop_joint_positions_deg}"
        )
        self.get_logger().info(
            f"Dice re-pick joint targets (deg): {self.dice_repick_joint_positions_deg}"
        )
        self.get_logger().info(f"Dice re-pick wait: {self.dice_repick_wait_sec:.2f} s")
        self.get_logger().info(
            f"Search start joint targets (deg): {self.search_start_joint_positions_deg}"
        )
        self.get_logger().info(
            f"Search look offsets: {len(self.search_look_offsets_deg)}; "
            f"extra search poses: {len(self.search_joint_positions_deg)}; "
            f"wait per pose: {self.search_pose_wait_sec:.2f} s"
        )
        self.get_logger().info(
            f"Approach-to-grasp wait: {self.approach_to_grasp_wait_sec:.2f} s"
        )
        self.get_logger().info(
            f"Pre-grasp clearance: {self.pre_grasp_clearance_z:.3f} m"
        )
        self.get_logger().info(
            "Grasp moves are slowed down: "
            f"velocity {self.grasp_velocity_scaling:.2f}, "
            f"acceleration {self.grasp_acceleration_scaling:.2f}."
        )
        self.get_logger().info(
            "Nearby approach search: "
            f"{'on' if self.approach_fallback_enabled else 'off'}, "
            f"xy step {self.approach_fallback_xy_step:.3f} m, "
            f"z step {self.approach_fallback_z_step:.3f} m."
        )
        self.get_logger().info(f"Gripper URScript target: {self.robot_ip}:30002")

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
                "MoveIt is not ready yet. Start move_group first or launch with launch_moveit:=true."
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
            f"{self.EXECUTION_TIMEOUT_SEC:.1f}s."
        )
        return False

    def _current_joint_positions(self, joint_names: list[str]) -> Optional[list[float]]:
        joint_state = self._moveit.joint_state
        if joint_state is None:
            self.get_logger().warn("No joint state. Using requested angles instead.")
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
                f"Missing joint state for {', '.join(missing_joint_names)}; "
                "using requested angles."
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
            f"{name}={math.degrees(value):.1f} deg"
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
            f"{name}={math.degrees(delta):.1f} deg" for name, delta in excessive
        )
        self.get_logger().debug(
            f"Skipping {label}; it would swing too far ({summary})."
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
            f"({age_sec:.2f}s old, max {self.MAX_POSE_AGE_SEC:.2f}s)."
        )
        return False

    def _execute_joint_configuration(
        self,
        label: str,
        joint_positions_rad: list[float],
        joint_positions_deg: list[float],
    ) -> tuple[bool, str]:
        if not self._moveit_ready():
            return False, "MoveIt planning service is not available."

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
                f"{name}={value:.1f} deg"
                for name, value in zip(joint_names, joint_positions_deg, strict=True)
            )
            self.get_logger().info(
                f"{label} using the nearest equivalent wrist angle: "
                f"[{requested_summary}] -> [{joints_summary}]"
            )

        self._publish_status(f"Moving to {label}: {joints_summary}")

        try:
            with self._motion_active_guard():
                self._moveit.move_to_configuration(
                    joint_positions=shortest_joint_positions_rad,
                    joint_names=joint_names,
                    tolerance=self.JOINT_TOLERANCE,
                )
                success = self._wait_for_motion_completion(label)
        except Exception as exc:
            return False, f"MoveIt failed during {label}: {exc}"

        if success:
            self._publish_status(f"{label.capitalize()} done.")
            return True, f"{label.capitalize()} done."
        return False, f"{label.capitalize()} failed."

    def _send_gripper_command(self, label: str) -> tuple[bool, str]:
        if label == "open":
            width = 36.9
            force = 80
        elif label == "close":
            width = 0.0
            force = 31
        else:
            return False, f"Unknown gripper command: {label}"

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
            return False, f"Failed to send gripper {label} command: {exc}"

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
                f"IK result missing {', '.join(missing_joint_names)}."
            )
            return None

        return [positions_by_name[name] for name in joint_names]

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
            self.get_logger().debug("Pose planning fallback timed out.")
            return None

        response = future.result()
        if response is None:
            self.get_logger().debug("Pose planning fallback returned no response.")
            return None

        motion_response = response.motion_plan_response
        if motion_response.error_code.val != self.MOVEIT_SUCCESS:
            self.get_logger().debug(
                "Pose planning fallback failed with MoveIt error code "
                f"{motion_response.error_code.val}."
            )
            return None

        trajectory = motion_response.trajectory.joint_trajectory
        if not trajectory.points:
            self.get_logger().debug("Pose planning fallback returned an empty path.")
            return None

        final_point = trajectory.points[-1]
        if len(trajectory.joint_names) != len(final_point.positions):
            self.get_logger().debug(
                "Pose planning fallback returned mismatched joint names and positions."
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
                "Pose planning fallback result missing "
                f"{', '.join(missing_joint_names)}."
            )
            return False

        joint_positions = [positions_by_name[name] for name in joint_names]
        shortest_joint_positions = self._nearest_equivalent_joint_positions(
            joint_positions, joint_names
        )
        self.get_logger().debug(
            "Planned pose joint target: "
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

        future = self._moveit.compute_ik_async(
            position=pose.pose.position,
            quat_xyzw=pose.pose.orientation,
            ik_link_name=self.TARGET_LINK,
            start_joint_state=self._moveit.joint_state,
            wait_for_server_timeout_sec=self.MOVEIT_WAIT_SECONDS,
        )
        if future is None:
            return False

        deadline = time.monotonic() + self.IK_WAIT_SECONDS
        while not future.done() and time.monotonic() < deadline:
            time.sleep(self.EXECUTION_POLL_INTERVAL_SEC)

        if not future.done():
            self.get_logger().warn(f"IK timeout after {self.IK_WAIT_SECONDS:.1f}s.")
            return False

        ik_joint_state = self._moveit.get_compute_ik_result(future)
        if ik_joint_state is None:
            return False

        joint_names = ur.joint_names(prefix="")
        ik_joint_positions = self._joint_positions_from_state(
            ik_joint_state, joint_names
        )
        if ik_joint_positions is None:
            return False

        shortest_joint_positions = self._nearest_equivalent_joint_positions(
            ik_joint_positions, joint_names
        )
        self.get_logger().debug(
            "IK joint target: "
            f"{self._format_joint_positions(joint_names, shortest_joint_positions)}"
        )
        if self._joint_target_has_excessive_motion(
            "IK pose target", joint_names, shortest_joint_positions
        ):
            return False

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
            return False, "MoveIt planning service is not available."
        if require_fresh_pose and not self._pose_is_fresh(received_ns, label):
            return (
                False,
                f"{label.capitalize()} pose is stale; reacquire the target first.",
            )

        candidates = (
            self._build_approach_candidates(pose)
            if label in {"approach", "camera-center approach"}
            else [(0.0, 0.0, 0.0, pose)]
        )

        last_failure_message = f"Could not move to {label}."

        try:
            with self._motion_active_guard():
                for dx, dy, dz, candidate_pose in candidates:
                    candidate_label = self._format_pose_candidate_label(dx, dy, dz)
                    planner_text = f" with {planner_label}" if planner_label else ""
                    if candidate_label:
                        self._publish_status(
                            f"Trying {label} nearby{candidate_label}{planner_text}: "
                            f"({candidate_pose.pose.position.x:.3f}, "
                            f"{candidate_pose.pose.position.y:.3f}, "
                            f"{candidate_pose.pose.position.z:.3f})"
                        )
                    else:
                        self._publish_status(
                            f"Moving to {label}{planner_text}: "
                            f"({candidate_pose.pose.position.x:.3f}, "
                            f"{candidate_pose.pose.position.y:.3f}, "
                            f"{candidate_pose.pose.position.z:.3f})"
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
                                "Quick IK failed for primary pose; "
                                "trying planned joint fallback."
                            )
                            used_nearest_ik = (
                                self._move_to_pose_with_planned_configuration(
                                    candidate_pose
                                )
                            )
                    if use_nearest_ik and not used_nearest_ik:
                        self.get_logger().debug(
                            f"No reasonable IK joint target for {label}"
                            f"{candidate_label}; trying pose goal."
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
                            "Trying pose goal."
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
                            f"(dx={dx:+.3f}, dy={dy:+.3f}, dz={dz:+.3f})."
                        )
                        return (
                            True,
                            f"{label.capitalize()} worked with nearby pose.",
                        )

                    last_failure_message = (
                        f"{label.capitalize()} failed{candidate_label}."
                    )
        except Exception as exc:
            return False, f"MoveIt failed during {label}: {exc}"

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
                "Using half-turn equivalent grasp yaw to minimize wrist rotation "
                f"({math.degrees(original_distance):.1f} deg -> "
                f"{math.degrees(flipped_distance):.1f} deg)."
            )
            self._align_quaternion_hemisphere(reference_pose, flipped_pose)
            return flipped_pose

        self._align_quaternion_hemisphere(reference_pose, original_pose)
        return original_pose

    @staticmethod
    def _format_pose_candidate_label(dx: float, dy: float, dz: float) -> str:
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            return ""
        return f" (dx={dx:+.3f}, dy={dy:+.3f}, dz={dz:+.3f})"

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
                "Grasp pose is stale; reacquire the target first.",
            )
        if not self._pose_is_fresh(self.latest_approach_received_ns, "approach"):
            return (
                None,
                None,
                "Approach pose is stale; reacquire the target first.",
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
            self._publish_status(f"Waiting {wait_sec:.2f}s {reason}.")

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

        return None, None, f"No updated pick pose received {reason}."

    def _wait_for_search_target(
        self, min_received_ns: int, label: str
    ) -> tuple[bool, str]:
        wait_sec = self.search_pose_wait_sec
        if wait_sec > 0.0:
            self._publish_status(
                f"Search workspace: waiting {wait_sec:.2f}s at {label}."
            )

        deadline = time.monotonic() + wait_sec
        while True:
            if (
                self.latest_approach_received_ns is not None
                and self.latest_grasp_received_ns is not None
                and self.latest_approach_received_ns > min_received_ns
                and self.latest_grasp_received_ns > min_received_ns
            ):
                approach_pose, grasp_pose, error_message = (
                    self._get_fresh_pick_pose_snapshot()
                )
                if approach_pose is not None and grasp_pose is not None:
                    return True, f"Target found from {label}."
                return False, error_message

            if wait_sec <= 0.0 or time.monotonic() >= deadline:
                break

            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0.0:
                break
            time.sleep(min(self.EXECUTION_POLL_INTERVAL_SEC, remaining_sec))

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
                return False, f"Workspace search failed moving to {label}: {message}"

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

            return self._execute_linear_pose("grasp", grasp_snapshot)

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

        ok, message = self._execute_grasp(approach_pose, grasp_pose)
        if not ok:
            return False, message

        return self._send_gripper_command("close")

    def _run_dice_test(self) -> tuple[bool, str]:
        while rclpy.ok() and not self._dice_test_stop_requested:
            self._publish_status("Dice test: picking.")

            ok, message = self._execute_pick_pipeline()
            if not ok:
                return (
                    False,
                    f"Dice test stopped during pick: {message}",
                )

            ok, message = self._execute_joint_configuration(
                "dice drop pose",
                self.dice_drop_joint_positions_rad,
                self.dice_drop_joint_positions_deg,
            )
            if not ok:
                return (
                    False,
                    f"Dice test stopped moving to drop pose: {message}",
                )

            ok, message = self._send_gripper_command("open")
            if not ok:
                return (
                    False,
                    f"Dice test stopped opening gripper: {message}",
                )

            ok, message = self._search_workspace()
            if not ok:
                return (
                    False,
                    f"Dice test stopped during workspace search: {message}",
                )

            if self.dice_repick_wait_sec > 0.0:
                self._publish_status(
                    f"Waiting {self.dice_repick_wait_sec:.2f}s before next dice pick."
                )
                time.sleep(self.dice_repick_wait_sec)

        if self._dice_test_stop_requested:
            return True, "Dice test stopped by request."
        return True, "Dice test stopped because ROS is shutting down."

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
        response.message = "Dice test stopping. waiting for current step to finish."
        return response

    def _handle_move_to_ready_pose(self, request, response):
        del request
        response.success, response.message = self._execute_joint_configuration(
            "ready pose",
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
