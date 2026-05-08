from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Optional

from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from moveit_msgs.srv import GetMotionPlan
from pymoveit2.moveit2 import MoveIt2State
from pymoveit2.robots import ur
from shape_msgs.msg import SolidPrimitive


class MotionExecutorMixin:
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
                "MoveIt is not ready yet. Start move group first, or launch with MoveIt enabled."
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
            f"{self.EXECUTION_TIMEOUT_SEC:.1f} seconds."
        )
        return False

    def _current_joint_positions(self, joint_names: list[str]) -> Optional[list[float]]:
        joint_state = self._moveit.joint_state
        if joint_state is None:
            self.get_logger().warn(
                "I do not have a joint state yet. Using the requested angles."
            )
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
                f"I am missing joint state for {', '.join(missing_joint_names)}. "
                "Using the requested angles."
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
            f"{name} {math.degrees(value):.1f} deg"
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
            f"{name} {math.degrees(delta):.1f} deg" for name, delta in excessive
        )
        self.get_logger().debug(f"Skipping {label}. It would move too far. {summary}.")
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
            f"at {age_sec:.2f} seconds old. Max is {self.MAX_POSE_AGE_SEC:.2f} seconds."
        )
        return False

    def _execute_joint_configuration(
        self,
        label: str,
        joint_positions_rad: list[float],
        joint_positions_deg: list[float],
    ) -> tuple[bool, str]:
        if not self._moveit_ready():
            return False, "MoveIt planning is not available."

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
                f"{name} {value:.1f} deg"
                for name, value in zip(joint_names, joint_positions_deg, strict=True)
            )
            self.get_logger().info(
                f"{label} is using the nearest matching wrist angle. "
                f"Requested {requested_summary}. Using {joints_summary}."
            )

        self._publish_status(f"Moving to {label}. {joints_summary}.")

        try:
            with self._motion_active_guard():
                self._moveit.move_to_configuration(
                    joint_positions=shortest_joint_positions_rad,
                    joint_names=joint_names,
                    tolerance=self.JOINT_TOLERANCE,
                )
                success = self._wait_for_motion_completion(label)
        except Exception as exc:
            return False, f"MoveIt had a problem during {label}. {exc}"

        if success:
            self._publish_status(f"{label.capitalize()} done.")
            return True, f"{label.capitalize()} done."
        return False, f"{label.capitalize()} did not finish."

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
                f"The IK result is missing {', '.join(missing_joint_names)}."
            )
            return None

        return [positions_by_name[name] for name in joint_names]

    def _ik_seed_candidates(
        self, joint_names: list[str]
    ) -> list[Optional[list[float]]]:
        current_positions = self._current_joint_positions(joint_names)
        if current_positions is None:
            return [None]
        if not self.prefer_elbow_up_ik:
            return [current_positions]

        try:
            elbow_index = joint_names.index("elbow_joint")
        except ValueError:
            return [current_positions]

        limits = self.JOINT_POSITION_LIMITS_RAD.get("elbow_joint")
        elbow_seed_values = [
            self.elbow_up_seed_rad,
            -self.elbow_up_seed_rad,
            0.0,
            current_positions[elbow_index],
        ]
        seeds: list[Optional[list[float]]] = []
        seen: set[tuple[int, ...]] = set()
        for elbow_seed in elbow_seed_values:
            seed_positions = list(current_positions)
            seed_positions[elbow_index] = self._nearest_equivalent_angle(
                elbow_seed,
                current_positions[elbow_index],
                limits,
            )
            key = tuple(round(value * 1000.0) for value in seed_positions)
            if key in seen:
                continue
            seen.add(key)
            seeds.append(seed_positions)
        return seeds or [current_positions]

    def _wait_for_future(self, future, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline:
            time.sleep(self.EXECUTION_POLL_INTERVAL_SEC)
        return future.done()

    def _compute_ik_joint_positions(
        self,
        pose: PoseStamped,
        start_joint_state,
        joint_names: list[str],
    ) -> Optional[list[float]]:
        future = self._moveit.compute_ik_async(
            position=pose.pose.position,
            quat_xyzw=pose.pose.orientation,
            ik_link_name=self.TARGET_LINK,
            start_joint_state=start_joint_state,
            wait_for_server_timeout_sec=self.MOVEIT_WAIT_SECONDS,
        )
        if future is None:
            return None
        if not self._wait_for_future(future, self.IK_WAIT_SECONDS):
            self.get_logger().warn(
                f"IK timed out after {self.IK_WAIT_SECONDS:.1f} seconds."
            )
            return None

        ik_joint_state = self._moveit.get_compute_ik_result(future)
        if ik_joint_state is None:
            return None

        ik_joint_positions = self._joint_positions_from_state(
            ik_joint_state, joint_names
        )
        if ik_joint_positions is None:
            return None

        shortest_joint_positions = self._nearest_equivalent_joint_positions(
            ik_joint_positions, joint_names
        )
        if self._joint_target_has_excessive_motion(
            "IK pose target", joint_names, shortest_joint_positions
        ):
            return None
        return shortest_joint_positions

    def _fk_link_z(
        self, joint_positions: list[float], link_name: str
    ) -> Optional[float]:
        future = self._moveit.compute_fk_async(
            joint_state=joint_positions,
            fk_link_names=[link_name],
        )
        if future is None:
            return None
        if not self._wait_for_future(future, self.IK_WAIT_SECONDS):
            self.get_logger().debug(
                f"FK timed out while checking {link_name} after {self.IK_WAIT_SECONDS:.1f} seconds."
            )
            return None

        poses = self._moveit.get_compute_fk_result(future, fk_link_names=[link_name])
        if poses is None:
            return None
        if not isinstance(poses, list):
            poses = [poses]
        if not poses:
            return None
        return float(poses[0].pose.position.z)

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
            self.get_logger().debug("The backup pose plan timed out.")
            return None

        response = future.result()
        if response is None:
            self.get_logger().debug("The backup pose plan gave no response.")
            return None

        motion_response = response.motion_plan_response
        if motion_response.error_code.val != self.MOVEIT_SUCCESS:
            self.get_logger().debug(
                "The backup pose plan failed with MoveIt error code "
                f"{motion_response.error_code.val}."
            )
            return None

        trajectory = motion_response.trajectory.joint_trajectory
        if not trajectory.points:
            self.get_logger().debug("The backup pose plan gave an empty path.")
            return None

        final_point = trajectory.points[-1]
        if len(trajectory.joint_names) != len(final_point.positions):
            self.get_logger().debug(
                "The backup pose plan gave joint names and positions that do not match."
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
                f"The backup pose plan is missing {', '.join(missing_joint_names)}."
            )
            return False

        joint_positions = [positions_by_name[name] for name in joint_names]
        shortest_joint_positions = self._nearest_equivalent_joint_positions(
            joint_positions, joint_names
        )
        self.get_logger().debug(
            "Planned pose joint target is "
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

        joint_names = ur.joint_names(prefix="")
        ik_candidates: list[tuple[list[float], Optional[float]]] = []
        for seed in self._ik_seed_candidates(joint_names):
            joint_positions = self._compute_ik_joint_positions(
                pose,
                seed,
                joint_names,
            )
            if joint_positions is None:
                continue

            elbow_height = (
                self._fk_link_z(joint_positions, self.ELBOW_HEIGHT_LINK_NAME)
                if self.prefer_elbow_up_ik
                else None
            )
            ik_candidates.append((joint_positions, elbow_height))

        if not ik_candidates:
            return False

        shortest_joint_positions, elbow_height = max(
            ik_candidates,
            key=lambda candidate: (
                candidate[1] if candidate[1] is not None else float("-inf")
            ),
        )
        self.get_logger().debug(
            "IK joint target is "
            f"{self._format_joint_positions(joint_names, shortest_joint_positions)}"
        )
        if elbow_height is not None:
            self.get_logger().debug(
                f"Selected IK with {self.ELBOW_HEIGHT_LINK_NAME} at "
                f"{elbow_height:.3f} m from {len(ik_candidates)} candidates."
            )

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
            return False, "MoveIt planning is not available."
        if require_fresh_pose and not self._pose_is_fresh(received_ns, label):
            return (
                False,
                f"{label.capitalize()} pose is too old. Reacquire the target first.",
            )

        candidates = (
            self._build_approach_candidates(pose)
            if label in {"approach", "camera-center approach"}
            else [(0.0, 0.0, 0.0, pose)]
        )

        last_failure_message = f"I could not move to {label}."

        try:
            with self._motion_active_guard():
                for dx, dy, dz, candidate_pose in candidates:
                    candidate_label = self._format_pose_candidate_label(dx, dy, dz)
                    planner_text = f" with {planner_label}" if planner_label else ""
                    if candidate_label:
                        self._publish_status(
                            f"Trying a nearby {label}{candidate_label}{planner_text}. "
                            f"{candidate_pose.pose.position.x:.3f}, "
                            f"{candidate_pose.pose.position.y:.3f}, "
                            f"{candidate_pose.pose.position.z:.3f}."
                        )
                    else:
                        self._publish_status(
                            f"Moving to {label}{planner_text}. "
                            f"{candidate_pose.pose.position.x:.3f}, "
                            f"{candidate_pose.pose.position.y:.3f}, "
                            f"{candidate_pose.pose.position.z:.3f}. "
                            f"Frame is {candidate_pose.header.frame_id or self.BASE_LINK_NAME}."
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
                                "Quick IK did not work for the main pose. "
                                "Trying a planned joint move."
                            )
                            used_nearest_ik = (
                                self._move_to_pose_with_planned_configuration(
                                    candidate_pose
                                )
                            )
                    if use_nearest_ik and not used_nearest_ik:
                        self.get_logger().debug(
                            f"No reasonable IK joint target for {label}"
                            f"{candidate_label}. Trying a pose goal."
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
                            "Trying a pose goal."
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
                            f"{dx:.3f}, {dy:.3f}, {dz:.3f}."
                        )
                        return (
                            True,
                            f"{label.capitalize()} worked with nearby pose.",
                        )

                    last_failure_message = (
                        f"{label.capitalize()} did not finish{candidate_label}."
                    )
        except Exception as exc:
            return False, f"MoveIt had a problem during {label}. {exc}"

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
