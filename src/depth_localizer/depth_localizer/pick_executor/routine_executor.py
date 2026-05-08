from __future__ import annotations

import time

import rclpy


class TestRoutinesMixin:
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
