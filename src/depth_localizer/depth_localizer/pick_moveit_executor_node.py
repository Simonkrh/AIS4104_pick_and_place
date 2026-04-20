#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetMotionPlan
from pymoveit2 import MoveIt2
from pymoveit2.robots import ur
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

class PickMoveItExecutorNode(Node):
    def __init__(self):
        super().__init__("pick_moveit_executor_node")

        self.declare_parameter("approach_topic", "/pick_approach_pose")
        self.declare_parameter("grasp_topic", "/pick_grasp_pose")
        self.declare_parameter("status_topic", "/pick_execution_status")
        self.declare_parameter("ur_type", "ur3e")
        self.declare_parameter("group_name", "ur_manipulator")
        self.declare_parameter("base_link_name", "")
        self.declare_parameter("end_effector_name", "")
        self.declare_parameter("target_link", "")
        self.declare_parameter("max_pose_age_sec", 2.0)
        self.declare_parameter("position_tolerance", 0.005)
        self.declare_parameter("orientation_tolerance", 0.05)
        self.declare_parameter("cartesian_grasp", False)
        self.declare_parameter("cartesian_max_step", 0.0025)
        self.declare_parameter("wait_for_moveit_seconds", 0.0)

        self.approach_topic = str(self.get_parameter("approach_topic").value)
        self.grasp_topic = str(self.get_parameter("grasp_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.ur_type = str(self.get_parameter("ur_type").value).strip() or "ur3e"
        self.group_name = str(self.get_parameter("group_name").value).strip() or "ur_manipulator"
        self.max_pose_age_sec = float(self.get_parameter("max_pose_age_sec").value)
        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        self.orientation_tolerance = float(
            self.get_parameter("orientation_tolerance").value
        )
        self.cartesian_grasp = bool(self.get_parameter("cartesian_grasp").value)
        self.cartesian_max_step = float(self.get_parameter("cartesian_max_step").value)
        self.wait_for_moveit_seconds = float(
            self.get_parameter("wait_for_moveit_seconds").value
        )

        base_link_name = str(self.get_parameter("base_link_name").value).strip()
        end_effector_name = str(self.get_parameter("end_effector_name").value).strip()
        target_link = str(self.get_parameter("target_link").value).strip()

        self.base_link_name = base_link_name or ur.base_link_name(prefix="")
        self.end_effector_name = end_effector_name or ur.end_effector_name(prefix="")
        self.target_link = target_link or self.end_effector_name

        self.callback_group = ReentrantCallbackGroup()
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.latest_approach_pose: Optional[PoseStamped] = None
        self.latest_grasp_pose: Optional[PoseStamped] = None
        self.latest_approach_received_ns: Optional[int] = None
        self.latest_grasp_received_ns: Optional[int] = None

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

        self._moveit = MoveIt2(
            node=self,
            joint_names=ur.joint_names(prefix=""),
            base_link_name=self.base_link_name,
            end_effector_name=self.end_effector_name,
            group_name=self.group_name,
            callback_group=self.callback_group,
            use_move_group_action=True,
        )

        self._plan_client = self.create_client(
            srv_type=GetMotionPlan,
            srv_name="plan_kinematic_path",
            callback_group=self.callback_group,
        )

        if self.wait_for_moveit_seconds > 0.0:
            self.get_logger().info(
                f"Waiting up to {self.wait_for_moveit_seconds:.1f}s for MoveIt planning service..."
            )
            self._plan_client.wait_for_service(timeout_sec=self.wait_for_moveit_seconds)

        self.get_logger().info(f"Subscribing approach pose: {self.approach_topic}")
        self.get_logger().info(f"Subscribing grasp pose: {self.grasp_topic}")
        self.get_logger().info(f"Publishing execution status: {self.status_topic}")
        self.get_logger().info("Services: ~/execute_approach, ~/execute_grasp, ~/execute_pick")
        self.get_logger().info(
            f"MoveIt group={self.group_name}, base={self.base_link_name}, tool={self.target_link}, ur_type={self.ur_type}"
        )

    def _on_approach_pose(self, msg: PoseStamped) -> None:
        self.latest_approach_pose = msg
        self.latest_approach_received_ns = self.get_clock().now().nanoseconds

    def _on_grasp_pose(self, msg: PoseStamped) -> None:
        self.latest_grasp_pose = msg
        self.latest_grasp_received_ns = self.get_clock().now().nanoseconds

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def _moveit_ready(self) -> bool:
        ready = self._plan_client.service_is_ready()
        if not ready:
            self.get_logger().warn(
                "MoveIt is not ready yet. Start move_group first or launch with launch_moveit:=true."
            )
        return ready

    def _pose_is_fresh(
        self, pose: PoseStamped, received_ns: Optional[int], label: str
    ) -> bool:
        if self.max_pose_age_sec <= 0.0:
            return True
        if received_ns is None:
            self.get_logger().warn(f"No local receipt timestamp recorded for {label} pose.")
            return False
        age_sec = (self.get_clock().now().nanoseconds - received_ns) / 1_000_000_000.0
        if age_sec <= self.max_pose_age_sec:
            return True
        self.get_logger().warn(
            f"Cached {label} pose is stale ({age_sec:.2f}s since receipt, limit {self.max_pose_age_sec:.2f}s)."
        )
        return False

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
        if not self._pose_is_fresh(pose, received_ns, label):
            return False, f"{label.capitalize()} pose is stale; reacquire the target first."

        self._publish_status(
            f"Planning {label} move to "
            f"({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}, {pose.pose.position.z:.3f}) "
            f"in {pose.header.frame_id}"
        )

        try:
            self._moveit.move_to_pose(
                pose=pose,
                target_link=self.target_link,
                tolerance_position=self.position_tolerance,
                tolerance_orientation=self.orientation_tolerance,
                cartesian=cartesian,
                cartesian_max_step=self.cartesian_max_step,
            )
            success = bool(self._moveit.wait_until_executed())
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
            cartesian=self.cartesian_grasp,
        )
        return response

    def _handle_execute_pick(self, request, response):
        del request
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

        ok, message = self._execute_pose(
            "grasp",
            self.latest_grasp_pose,
            self.latest_grasp_received_ns,
            cartesian=self.cartesian_grasp,
        )
        response.success = ok
        response.message = message
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
