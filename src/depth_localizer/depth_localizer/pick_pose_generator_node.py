#!/usr/bin/env python3
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


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


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PickPoseGeneratorNode(Node):
    APPROACH_TF_FRAME = "pick_approach"
    GRASP_TF_FRAME = "pick_grasp"

    def __init__(self):
        super().__init__("pick_pose_generator_node")

        self.declare_parameter("input_topic", "/pick_target_pose")
        self.declare_parameter("approach_topic", "/pick_approach_pose")
        self.declare_parameter("grasp_topic", "/pick_grasp_pose")
        self.declare_parameter("approach_offset_z", 0.35)
        self.declare_parameter("grasp_offset_z", -0.02)
        self.declare_parameter("table_top_z", -0.01)
        self.declare_parameter("min_grasp_clearance_z", 0.005)
        self.declare_parameter("tool_roll", math.pi)
        self.declare_parameter("tool_yaw", math.pi)
        self.declare_parameter("approach_camera_offset_y", 0.10)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.approach_topic = str(self.get_parameter("approach_topic").value)
        self.grasp_topic = str(self.get_parameter("grasp_topic").value)
        self.approach_offset_z = float(self.get_parameter("approach_offset_z").value)
        self.grasp_offset_z = float(self.get_parameter("grasp_offset_z").value)
        self.table_top_z = float(self.get_parameter("table_top_z").value)
        self.min_grasp_clearance_z = max(
            float(self.get_parameter("min_grasp_clearance_z").value), 0.0
        )
        self.tool_roll = float(self.get_parameter("tool_roll").value)
        self.tool_yaw = float(self.get_parameter("tool_yaw").value)
        self.approach_camera_offset_y = float(
            self.get_parameter("approach_camera_offset_y").value
        )

        self.approach_pub = self.create_publisher(PoseStamped, self.approach_topic, 10)
        self.grasp_pub = self.create_publisher(PoseStamped, self.grasp_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.sub = self.create_subscription(
            PoseStamped, self.input_topic, self.on_target, 10
        )

        self.get_logger().info(f"Subscribing target pose: {self.input_topic}")
        self.get_logger().info(f"Publishing approach pose: {self.approach_topic}")
        self.get_logger().info(f"Publishing grasp pose: {self.grasp_topic}")

    def on_target(self, msg: PoseStamped):
        output_stamp = self.get_clock().now().to_msg()

        approach_pose = self._make_pose(
            msg,
            self.approach_offset_z,
            output_stamp,
            compensate_camera_offset=True,
            apply_object_yaw=False,
        )
        grasp_pose = self._make_pose(
            msg,
            self.grasp_offset_z,
            output_stamp,
            compensate_camera_offset=False,
            apply_object_yaw=True,
        )

        self.approach_pub.publish(approach_pose)
        self.grasp_pub.publish(grasp_pose)

        self.tf_broadcaster.sendTransform(
            self._pose_to_transform(approach_pose, self.APPROACH_TF_FRAME)
        )
        self.tf_broadcaster.sendTransform(
            self._pose_to_transform(grasp_pose, self.GRASP_TF_FRAME)
        )

    def _make_pose(
        self,
        source_pose: PoseStamped,
        z_offset: float,
        output_stamp,
        compensate_camera_offset: bool,
        apply_object_yaw: bool,
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = output_stamp
        pose.header.frame_id = source_pose.header.frame_id
        source_delta_yaw = 0.0
        if apply_object_yaw:
            source_delta_yaw = _yaw_from_quaternion(
                float(source_pose.pose.orientation.x),
                float(source_pose.pose.orientation.y),
                float(source_pose.pose.orientation.z),
                float(source_pose.pose.orientation.w),
            )
        orientation_xyzw = _quaternion_from_rpy(
            self.tool_roll,
            0.0,
            self.tool_yaw + source_delta_yaw,
        )
        pose.pose.position.x = float(source_pose.pose.position.x)
        pose.pose.position.y = float(source_pose.pose.position.y)
        pose.pose.position.z = float(source_pose.pose.position.z) + z_offset
        if not compensate_camera_offset:
            min_grasp_z = self.table_top_z + self.min_grasp_clearance_z
            pose.pose.position.z = max(pose.pose.position.z, min_grasp_z)
        if compensate_camera_offset:
            offset_x, offset_y = self._rotate_y_offset_to_pose_frame(
                self.approach_camera_offset_y,
                orientation_xyzw,
            )
            pose.pose.position.x -= offset_x
            pose.pose.position.y -= offset_y
        pose.pose.orientation.x = float(orientation_xyzw[0])
        pose.pose.orientation.y = float(orientation_xyzw[1])
        pose.pose.orientation.z = float(orientation_xyzw[2])
        pose.pose.orientation.w = float(orientation_xyzw[3])
        return pose

    def _rotate_y_offset_to_pose_frame(
        self,
        local_y: float,
        orientation_xyzw: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        qx, qy, qz, qw = orientation_xyzw
        xx = qx * qx
        xy = qx * qy
        zz = qz * qz
        zw = qz * qw

        rot_xy = 2.0 * (xy - zw)
        rot_yy = 1.0 - 2.0 * (xx + zz)

        return (
            rot_xy * local_y,
            rot_yy * local_y,
        )

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
