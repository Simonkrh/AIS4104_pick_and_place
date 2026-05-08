from __future__ import annotations

import time
from typing import Optional

from geometry_msgs.msg import PoseStamped


class PickPipelineMixin:
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
