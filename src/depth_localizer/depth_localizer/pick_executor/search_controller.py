from __future__ import annotations

import math
import time


class SearchControllerMixin:
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
