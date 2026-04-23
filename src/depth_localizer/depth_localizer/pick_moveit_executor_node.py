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
from moveit_msgs.srv import GetMotionPlan
from pymoveit2 import MoveIt2
from pymoveit2.moveit2 import MoveIt2State
from pymoveit2.robots import ur
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
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
    CARTESIAN_GRASP = False
    CARTESIAN_MAX_STEP = 0.0025
    MOVEIT_WAIT_SECONDS = 5.0
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
        self.declare_parameter(
            "ready_joint_positions_deg",
            [-90.0, -90.0, 0.0, -180.0, 90.0, 180.0],
        )
        self.declare_parameter("robot_ip", self.DEFAULT_ROBOT_IP)

        self.approach_topic = str(self.get_parameter("approach_topic").value)
        self.grasp_topic = str(self.get_parameter("grasp_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.motion_active_topic = str(
            self.get_parameter("motion_active_topic").value
        )
        self.approach_to_grasp_wait_sec = max(
            float(self.get_parameter("approach_to_grasp_wait_sec").value), 0.0
        )
        self.robot_ip = str(self.get_parameter("robot_ip").value).strip()
        self.ready_joint_positions_deg = self._parse_joint_positions_deg(
            self.get_parameter("ready_joint_positions_deg").value
        )
        self.ready_joint_positions_rad = [
            math.radians(value) for value in self.ready_joint_positions_deg
        ]

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

        self._plan_client = self.create_client(
            srv_type=GetMotionPlan,
            srv_name="plan_kinematic_path",
            callback_group=self.callback_group,
        )

        self.get_logger().info(
            f"Waiting up to {self.MOVEIT_WAIT_SECONDS:.1f}s for MoveIt planning service..."
        )
        self._plan_client.wait_for_service(timeout_sec=self.MOVEIT_WAIT_SECONDS)

        self.get_logger().info(f"Subscribing approach pose: {self.approach_topic}")
        self.get_logger().info(f"Subscribing grasp pose: {self.grasp_topic}")
        self.get_logger().info(f"Publishing execution status: {self.status_topic}")
        self.get_logger().info(
            f"Publishing motion active flag: {self.motion_active_topic}"
        )
        self.get_logger().info(
            "Services: ~/execute_approach, ~/execute_grasp, ~/execute_pick, "
            "~/move_to_ready_pose, ~/move_to_start_pose, "
            "~/open_gripper, ~/close_gripper"
        )
        self.get_logger().info(
            f"MoveIt group={self.GROUP_NAME}, base={self.BASE_LINK_NAME}, tool={self.TARGET_LINK}"
        )
        self.get_logger().info(
            f"Ready pose joint targets (deg): {self.ready_joint_positions_deg}"
        )
        self.get_logger().info(
            f"Approach-to-grasp wait: {self.approach_to_grasp_wait_sec:.2f} s"
        )
        self.get_logger().info(f"Gripper URScript target: {self.robot_ip}:30002")

    def _parse_joint_positions_deg(self, value) -> list[float]:
        if isinstance(value, str):
            try:
                parsed_value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    "ready_joint_positions_deg must be a list string like "
                    '"[-90.0, -90.0, 0.0, -180.0, 90.0, 180.0]".'
                ) from exc
        else:
            parsed_value = value

        if not isinstance(parsed_value, (list, tuple)):
            raise ValueError("ready_joint_positions_deg must contain six joint values.")

        joint_positions = [float(item) for item in parsed_value]
        expected_joint_count = len(ur.joint_names(prefix=""))
        if len(joint_positions) != expected_joint_count:
            raise ValueError(
                "ready_joint_positions_deg must contain "
                f"{expected_joint_count} values in UR order "
                "[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]."
            )

        return joint_positions

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

    def _pose_is_fresh(self, received_ns: Optional[int], label: str) -> bool:
        if received_ns is None:
            self.get_logger().warn(
                f"No local receipt timestamp recorded for {label} pose."
            )
            return False
        age_sec = (self.get_clock().now().nanoseconds - received_ns) / 1_000_000_000.0
        if age_sec <= self.MAX_POSE_AGE_SEC:
            return True
        self.get_logger().warn(
            f"Cached {label} pose is stale "
            f"({age_sec:.2f}s since receipt, limit {self.MAX_POSE_AGE_SEC:.2f}s)."
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
        joints_summary = ", ".join(
            f"{name}={value:.1f} deg"
            for name, value in zip(joint_names, joint_positions_deg, strict=True)
        )
        self._publish_status(f"Planning {label} joint move: {joints_summary}")

        try:
            with self._motion_active_guard():
                self._moveit.move_to_configuration(
                    joint_positions=joint_positions_rad,
                    joint_names=joint_names,
                    tolerance=self.JOINT_TOLERANCE,
                )
                success = self._wait_for_motion_completion(label)
        except Exception as exc:
            return False, f"MoveIt failed during {label}: {exc}"

        if success:
            self._publish_status(f"{label.capitalize()} joint move completed.")
            return True, f"{label.capitalize()} joint move completed."
        return False, f"{label.capitalize()} joint move failed."

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

        message = f"Sent gripper {label} command."
        self._publish_status(message)
        time.sleep(self.GRIPPER_SETTLE_SEC)
        return True, message

    def _execute_pose(
        self,
        label: str,
        pose: Optional[PoseStamped],
        received_ns: Optional[int],
        cartesian: bool = False,
    ) -> tuple[bool, str]:
        if pose is None:
            return False, f"No cached {label} pose yet."
        if not self._moveit_ready():
            return False, "MoveIt planning service is not available."
        if not self._pose_is_fresh(received_ns, label):
            return (
                False,
                f"{label.capitalize()} pose is stale; reacquire the target first.",
            )

        self._publish_status(
            f"Planning {label} move to "
            f"({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f}) "
            f"in {pose.header.frame_id}"
        )

        try:
            with self._motion_active_guard():
                self._moveit.move_to_pose(
                    pose=pose,
                    target_link=self.TARGET_LINK,
                    tolerance_position=self.POSITION_TOLERANCE,
                    tolerance_orientation=self.ORIENTATION_TOLERANCE,
                    cartesian=cartesian,
                    cartesian_max_step=self.CARTESIAN_MAX_STEP,
                )
                success = self._wait_for_motion_completion(label)
        except Exception as exc:
            return False, f"MoveIt failed during {label}: {exc}"

        if success:
            self._publish_status(f"{label.capitalize()} move completed.")
            return True, f"{label.capitalize()} move completed."
        return False, f"{label.capitalize()} move failed."

    def _handle_execute_approach(self, request, response):
        del request
        response.success, response.message = self._execute_pose(
            "approach",
            self.latest_approach_pose,
            self.latest_approach_received_ns,
            cartesian=False,
        )
        return response

    def _handle_execute_grasp(self, request, response):
        del request
        response.success, response.message = self._execute_pose(
            "grasp",
            self.latest_grasp_pose,
            self.latest_grasp_received_ns,
            cartesian=self.CARTESIAN_GRASP,
        )
        return response

    def _handle_execute_pick(self, request, response):
        del request
        ok, message = self._send_gripper_command("open")
        if not ok:
            response.success = False
            response.message = message
            return response

        ok, message = self._execute_pose(
            "approach",
            self.latest_approach_pose,
            self.latest_approach_received_ns,
            cartesian=False,
        )
        if not ok:
            response.success = False
            response.message = message
            return response

        if self.approach_to_grasp_wait_sec > 0.0:
            self._publish_status(
                f"Waiting {self.approach_to_grasp_wait_sec:.2f}s to reacquire target before grasp."
            )
            time.sleep(self.approach_to_grasp_wait_sec)

        ok, message = self._execute_pose(
            "grasp",
            self.latest_grasp_pose,
            self.latest_grasp_received_ns,
            cartesian=self.CARTESIAN_GRASP,
        )
        if not ok:
            response.success = False
            response.message = message
            return response

        ok, message = self._send_gripper_command("close")
        response.success = ok
        response.message = message
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
