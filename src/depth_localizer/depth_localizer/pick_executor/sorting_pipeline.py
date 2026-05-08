from __future__ import annotations

import time
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped


class SortingPipelineMixin:
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
