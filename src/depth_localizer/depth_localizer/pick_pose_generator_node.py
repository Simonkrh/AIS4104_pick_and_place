#!/usr/bin/env python3
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def _stamp_to_nanoseconds(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _quaternion_from_rpy(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


class PickPoseGeneratorNode(Node):
    def __init__(self):
        super().__init__("pick_pose_generator_node")

        self.declare_parameter("input_topic", "/pick_target_pose")
        self.declare_parameter("approach_topic", "/pick_approach_pose")
        self.declare_parameter("grasp_topic", "/pick_grasp_pose")
        self.declare_parameter("approach_offset_z", 0.10)
        self.declare_parameter("grasp_offset_z", 0.02)
        self.declare_parameter("tool_roll", math.pi)
        self.declare_parameter("tool_pitch", 0.0)
        self.declare_parameter("tool_yaw", 0.0)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("approach_tf_child_frame", "pick_approach")
        self.declare_parameter("grasp_tf_child_frame", "pick_grasp")
        self.declare_parameter("max_input_stamp_age_sec", 2.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.approach_topic = str(self.get_parameter("approach_topic").value)
        self.grasp_topic = str(self.get_parameter("grasp_topic").value)
        self.approach_offset_z = float(self.get_parameter("approach_offset_z").value)
        self.grasp_offset_z = float(self.get_parameter("grasp_offset_z").value)
        tool_roll = float(self.get_parameter("tool_roll").value)
        tool_pitch = float(self.get_parameter("tool_pitch").value)
        tool_yaw = float(self.get_parameter("tool_yaw").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.approach_tf_child_frame = str(
            self.get_parameter("approach_tf_child_frame").value
        ).strip()
        self.grasp_tf_child_frame = str(
            self.get_parameter("grasp_tf_child_frame").value
        ).strip()
        self.max_input_stamp_age_sec = float(
            self.get_parameter("max_input_stamp_age_sec").value
        )

        self.orientation_xyzw = _quaternion_from_rpy(tool_roll, tool_pitch, tool_yaw)

        self.approach_pub = self.create_publisher(PoseStamped, self.approach_topic, 10)
        self.grasp_pub = self.create_publisher(PoseStamped, self.grasp_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None

        self.sub = self.create_subscription(
            PoseStamped, self.input_topic, self.on_target, 10
        )

        self._last_stamp_warn_ns = 0

        self.get_logger().info(f"Subscribing target pose: {self.input_topic}")
        self.get_logger().info(f"Publishing approach pose: {self.approach_topic}")
        self.get_logger().info(f"Publishing grasp pose: {self.grasp_topic}")

    def _is_stamp_stale(self, stamp) -> bool:
        stamp_ns = _stamp_to_nanoseconds(stamp)
        if stamp_ns <= 0:
            return True
        now_ns = self.get_clock().now().nanoseconds
        max_age_ns = int(self.max_input_stamp_age_sec * 1e9)
        return abs(now_ns - stamp_ns) > max_age_ns

    def _warn_stale_stamp_once(self, stamp) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_stamp_warn_ns > 1_000_000_000:
            self._last_stamp_warn_ns = now_ns
            age_sec = (now_ns - _stamp_to_nanoseconds(stamp)) / 1e9
            self.get_logger().warn(
                f"Incoming target pose timestamp is stale by {age_sec:.3f}s; "
                "publishing current-time approach/grasp poses and TF."
            )

    def on_target(self, msg: PoseStamped):
        if self._is_stamp_stale(msg.header.stamp):
            self._warn_stale_stamp_once(msg.header.stamp)

        output_stamp = self.get_clock().now().to_msg()

        approach_pose = self._make_pose(msg, self.approach_offset_z, output_stamp)
        grasp_pose = self._make_pose(msg, self.grasp_offset_z, output_stamp)

        self.approach_pub.publish(approach_pose)
        self.grasp_pub.publish(grasp_pose)

        if self.tf_broadcaster is not None:
            if self.approach_tf_child_frame:
                self.tf_broadcaster.sendTransform(
                    self._pose_to_transform(approach_pose, self.approach_tf_child_frame)
                )
            if self.grasp_tf_child_frame:
                self.tf_broadcaster.sendTransform(
                    self._pose_to_transform(grasp_pose, self.grasp_tf_child_frame)
                )

    def _make_pose(
        self, source_pose: PoseStamped, z_offset: float, output_stamp
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = output_stamp
        pose.header.frame_id = source_pose.header.frame_id
        pose.pose.position.x = float(source_pose.pose.position.x)
        pose.pose.position.y = float(source_pose.pose.position.y)
        pose.pose.position.z = float(source_pose.pose.position.z) + z_offset
        pose.pose.orientation.x = float(self.orientation_xyzw[0])
        pose.pose.orientation.y = float(self.orientation_xyzw[1])
        pose.pose.orientation.z = float(self.orientation_xyzw[2])
        pose.pose.orientation.w = float(self.orientation_xyzw[3])
        return pose

    @staticmethod
    def _pose_to_transform(pose: PoseStamped, child_frame: str) -> TransformStamped:
        transform = TransformStamped()
        transform.header = pose.header
        transform.child_frame_id = child_frame
        transform.transform.translation.x = float(pose.pose.position.x)
        transform.transform.translation.y = float(pose.pose.position.y)
        transform.transform.translation.z = float(pose.pose.position.z)
        transform.transform.rotation = pose.pose.orientation
        return transform


def main():
    rclpy.init()
    node = PickPoseGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
