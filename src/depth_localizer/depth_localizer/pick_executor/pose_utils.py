from __future__ import annotations

import math

from geometry_msgs.msg import PoseStamped


class PoseUtilsMixin:
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
    def _quaternion_from_rotation_vector(
        rx: float, ry: float, rz: float
    ) -> tuple[float, float, float, float]:
        angle = math.sqrt(rx * rx + ry * ry + rz * rz)
        if angle <= 1e-9:
            return 0.0, 0.0, 0.0, 1.0

        scale = math.sin(0.5 * angle) / angle
        return rx * scale, ry * scale, rz * scale, math.cos(0.5 * angle)

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
                "Using the half turn grasp yaw to reduce wrist rotation. "
                f"{math.degrees(original_distance):.1f} deg to "
                f"{math.degrees(flipped_distance):.1f} deg."
            )
            self._align_quaternion_hemisphere(reference_pose, flipped_pose)
            return flipped_pose

        self._align_quaternion_hemisphere(reference_pose, original_pose)
        return original_pose

    @staticmethod
    def _format_pose_candidate_label(dx: float, dy: float, dz: float) -> str:
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            return ""
        return f" with offset {dx:.3f}, {dy:.3f}, {dz:.3f}"

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
