#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def matrix_from_rt(rotation_matrix: np.ndarray, translation_xyz: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation_xyz, dtype=np.float64).reshape(3)
    return matrix


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def pose_dict_from_matrix(matrix: np.ndarray) -> dict:
    return {
        "translation_xyz": matrix[:3, 3].astype(float).tolist(),
        "quaternion_xyzw": Rotation.from_matrix(matrix[:3, :3]).as_quat().astype(float).tolist(),
    }


def matrix_from_pose_dict(pose: dict) -> np.ndarray:
    return matrix_from_rt(
        Rotation.from_quat(pose["quaternion_xyzw"]).as_matrix(),
        np.asarray(pose["translation_xyz"], dtype=np.float64),
    )


def rvec_tvec_to_matrix(rvec, tvec) -> np.ndarray:
    return matrix_from_rt(
        Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64).reshape(3)).as_matrix(),
        np.asarray(tvec, dtype=np.float64).reshape(3),
    )


def transform_msg_to_matrix(transform_msg) -> np.ndarray:
    translation = np.array(
        [
            transform_msg.translation.x,
            transform_msg.translation.y,
            transform_msg.translation.z,
        ],
        dtype=np.float64,
    )
    rotation = Rotation.from_quat(
        [
            transform_msg.rotation.x,
            transform_msg.rotation.y,
            transform_msg.rotation.z,
            transform_msg.rotation.w,
        ]
    ).as_matrix()
    return matrix_from_rt(rotation, translation)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict):
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_object_points(board_cols: int, board_rows: int, square_size_m: float) -> np.ndarray:
    points = np.zeros((board_rows * board_cols, 3), dtype=np.float32)
    grid = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    points[:, :2] = grid * square_size_m
    return points


def evaluate_solution(samples: list[dict], tool_t_camera: np.ndarray) -> dict:
    base_t_targets = []
    sample_refs = []
    for sample in samples:
        base_t_tool = matrix_from_pose_dict(sample["base_T_tool"])
        target_to_camera = rvec_tvec_to_matrix(
            sample["target_to_camera"]["rvec"],
            sample["target_to_camera"]["tvec_m"],
        )
        base_t_target = base_t_tool @ tool_t_camera @ target_to_camera
        base_t_targets.append(base_t_target)
        sample_refs.append(
            {
                "index": int(sample.get("index", len(sample_refs))),
                "captured_utc": sample.get("captured_utc"),
                "image_path": sample.get("image_path"),
                "overlay_image_path": sample.get("overlay_image_path"),
            }
        )

    translations = np.asarray([matrix[:3, 3] for matrix in base_t_targets], dtype=np.float64)
    translation_mean = translations.mean(axis=0)
    translation_std = translations.std(axis=0)

    rotations = Rotation.from_matrix([matrix[:3, :3] for matrix in base_t_targets])
    mean_rotation = rotations.mean()
    angle_errors_deg = ((mean_rotation.inv() * rotations).magnitude() * 180.0 / math.pi)
    translation_error_vectors = translations - translation_mean
    translation_errors_m = np.linalg.norm(translation_error_vectors, axis=1)

    position_std_norm_m = float(np.linalg.norm(translation_std))
    orientation_std_deg = float(angle_errors_deg.std())
    position_scale = max(position_std_norm_m, 1e-9)
    orientation_scale = max(orientation_std_deg, 1e-9)

    sample_errors = []
    for sample_ref, translation_error_vector, translation_error_m, orientation_error_deg in zip(
        sample_refs,
        translation_error_vectors,
        translation_errors_m,
        angle_errors_deg,
    ):
        ranking_score = math.hypot(
            float(translation_error_m) / position_scale,
            float(orientation_error_deg) / orientation_scale,
        )
        sample_errors.append(
            {
                **sample_ref,
                "translation_error_xyz_m": translation_error_vector.astype(float).tolist(),
                "translation_error_m": float(translation_error_m),
                "orientation_error_deg": float(orientation_error_deg),
                "ranking_score": float(ranking_score),
            }
        )

    sample_errors.sort(
        key=lambda item: (
            item["ranking_score"],
            item["orientation_error_deg"],
            item["translation_error_m"],
        ),
        reverse=True,
    )

    return {
        "sample_count": len(samples),
        "target_position_std_m": translation_std.astype(float).tolist(),
        "target_position_std_norm_m": position_std_norm_m,
        "target_orientation_std_deg": orientation_std_deg,
        "target_orientation_max_error_deg": float(angle_errors_deg.max()),
        "sample_errors": sample_errors,
    }


def validate_solution(samples: list[dict], tool_t_camera: np.ndarray) -> dict:
    evaluation = evaluate_solution(samples, tool_t_camera)
    evaluation.pop("sample_errors", None)
    return evaluation
