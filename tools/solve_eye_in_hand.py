#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from handeye_utils import (
    METHODS,
    load_json,
    matrix_from_pose_dict,
    matrix_from_rt,
    pose_dict_from_matrix,
    rvec_tvec_to_matrix,
    save_json,
    validate_solution,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Solve eye-in-hand calibration from saved samples.")
    parser.add_argument("--session-dir", default="calibration/eye_in_hand")
    return parser.parse_args()


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)
    sample_file = session_dir / "samples.json"
    if not sample_file.exists():
        raise FileNotFoundError(f"Missing sample file: {sample_file}")

    payload = load_json(sample_file)
    samples = payload.get("samples", [])
    if len(samples) < 5:
        raise RuntimeError("Need at least 5 samples. Aim for 15-30 good poses.")

    rotations_gripper_to_base = []
    translations_gripper_to_base = []
    rotations_target_to_camera = []
    translations_target_to_camera = []

    for sample in samples:
        base_t_tool = matrix_from_pose_dict(sample["base_T_tool"])
        target_t_camera = rvec_tvec_to_matrix(
            sample["target_to_camera"]["rvec"],
            sample["target_to_camera"]["tvec_m"],
        )
        rotations_gripper_to_base.append(base_t_tool[:3, :3])
        translations_gripper_to_base.append(base_t_tool[:3, 3])
        rotations_target_to_camera.append(target_t_camera[:3, :3])
        translations_target_to_camera.append(target_t_camera[:3, 3])

    results = []
    failures = []
    for method_name, method in METHODS.items():
        try:
            rotation_camera_to_gripper, translation_camera_to_gripper = cv2.calibrateHandEye(
                rotations_gripper_to_base,
                translations_gripper_to_base,
                rotations_target_to_camera,
                translations_target_to_camera,
                method=method,
            )
        except cv2.error as exc:
            failures.append({"method": method_name, "error": str(exc)})
            continue

        tool_t_camera = matrix_from_rt(
            rotation_camera_to_gripper,
            np.asarray(translation_camera_to_gripper, dtype=np.float64).reshape(3),
        )
        results.append(
            {
                "method": method_name,
                "tool_T_camera_frame": pose_dict_from_matrix(tool_t_camera),
                "validation": validate_solution(samples, tool_t_camera),
            }
        )

    if not results:
        raise RuntimeError("All OpenCV hand-eye methods failed. The sample set is likely poor.")

    best_result = min(
        results,
        key=lambda item: (
            item["validation"]["target_position_std_norm_m"],
            item["validation"]["target_orientation_std_deg"],
        ),
    )

    output_file = session_dir / "handeye_result.json"
    save_json(
        output_file,
        {
            "metadata": payload.get("metadata", {}),
            "best_result": best_result,
            "all_results": results,
            "failed_methods": failures,
        },
    )

    print(f"Saved result to {output_file}")
    print(f"Best method: {best_result['method']}")
    print("tool -> camera_frame:", best_result["tool_T_camera_frame"])
    print(
        "target position scatter norm (m):",
        round(best_result["validation"]["target_position_std_norm_m"], 6),
    )
    print(
        "target orientation std (deg):",
        round(best_result["validation"]["target_orientation_std_deg"], 6),
    )


if __name__ == "__main__":
    main()
