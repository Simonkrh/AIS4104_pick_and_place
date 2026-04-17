#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from handeye_utils import (
    METHODS,
    evaluate_solution,
    load_json,
    matrix_from_pose_dict,
    matrix_from_rt,
    pose_dict_from_matrix,
    rvec_tvec_to_matrix,
    save_json,
)


DEFAULT_SESSION_DIR = "calibration/eye_in_hand_charuco"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Solve eye-in-hand calibration from saved samples. Defaults are tuned "
            "for the current ChArUco session."
        )
    )
    parser.add_argument(
        "--session-dir",
        default=DEFAULT_SESSION_DIR,
        help="Session folder to solve. Defaults to the current ChArUco session path.",
    )
    parser.add_argument(
        "--report-top",
        type=int,
        default=10,
        help="Print the worst N samples ranked by per-sample inconsistency for the best method.",
    )
    parser.add_argument(
        "--prune-worst",
        type=int,
        default=0,
        help="Remove the worst N ranked samples based on the best method.",
    )
    parser.add_argument(
        "--prune-output",
        default="",
        help="Path to write the pruned samples JSON. Defaults to <session-dir>/samples_pruned.json.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite samples.json when pruning. A .bak backup is created first.",
    )
    return parser.parse_args()


def sample_identity(sample: dict) -> tuple[int, str, str]:
    return (
        int(sample.get("index", -1)),
        str(sample.get("image_path", "")),
        str(sample.get("captured_utc", "")),
    )


def ranking_identity(sample_error: dict) -> tuple[int, str, str]:
    return (
        int(sample_error.get("index", -1)),
        str(sample_error.get("image_path", "")),
        str(sample_error.get("captured_utc", "")),
    )


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

    board = payload.get("metadata", {}).get("board", {})
    board_type = str(board.get("type", "chessboard"))
    board_cols = int(board.get("cols", 0) or 0)
    board_rows = int(board.get("rows", 0) or 0)
    if (
        board_type == "chessboard"
        and board_cols > 0
        and board_rows > 0
        and board_cols % 2 == 0
        and board_rows % 2 == 0
    ):
        print(
            "Warning: the sample set uses an even-by-even checkerboard "
            f"({board_cols}x{board_rows} inner corners). Plain chessboards with this "
            "geometry can produce ambiguous corner ordering and 180-degree-like pose flips."
        )

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
        evaluation = evaluate_solution(samples, tool_t_camera)
        results.append(
            {
                "method": method_name,
                "tool_T_camera_frame": pose_dict_from_matrix(tool_t_camera),
                "validation": {
                    key: value
                    for key, value in evaluation.items()
                    if key != "sample_errors"
                },
                "sample_ranking": evaluation["sample_errors"],
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
    best_sample_ranking = best_result["sample_ranking"]

    serializable_results = []
    for item in results:
        serializable_results.append(
            {
                "method": item["method"],
                "tool_T_camera_frame": item["tool_T_camera_frame"],
                "validation": item["validation"],
            }
        )

    output_file = session_dir / "handeye_result.json"
    save_json(
        output_file,
        {
            "metadata": payload.get("metadata", {}),
            "best_result": {
                "method": best_result["method"],
                "tool_T_camera_frame": best_result["tool_T_camera_frame"],
                "validation": best_result["validation"],
            },
            "all_results": serializable_results,
            "failed_methods": failures,
            "sample_ranking": {
                "method": best_result["method"],
                "ranking_basis": (
                    "Samples are ranked by how inconsistent their reconstructed target pose is "
                    "in the base frame, using the best hand-eye solution."
                ),
                "samples": best_sample_ranking,
            },
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
    report_top = max(0, args.report_top)
    if report_top:
        print(f"Worst {min(report_top, len(best_sample_ranking))} samples:")
        for rank, sample_error in enumerate(best_sample_ranking[:report_top], start=1):
            print(
                f"  {rank}. index={sample_error['index']} "
                f"score={sample_error['ranking_score']:.3f} "
                f"trans_err_m={sample_error['translation_error_m']:.4f} "
                f"rot_err_deg={sample_error['orientation_error_deg']:.2f} "
                f"path={sample_error.get('image_path')}"
            )

    prune_worst = max(0, args.prune_worst)
    if prune_worst:
        if len(samples) - prune_worst < 5:
            raise RuntimeError(
                "Pruning would leave fewer than 5 samples, which is too few to calibrate."
            )

        removal_keys = {
            ranking_identity(sample_error)
            for sample_error in best_sample_ranking[:prune_worst]
        }
        kept_samples = [
            sample for sample in samples if sample_identity(sample) not in removal_keys
        ]

        pruned_payload = {
            "metadata": payload.get("metadata", {}),
            "samples": kept_samples,
        }
        if args.in_place:
            backup_file = sample_file.with_suffix(sample_file.suffix + ".bak")
            save_json(backup_file, payload)
            save_json(sample_file, pruned_payload)
            print(f"Created backup at {backup_file}")
            print(f"Pruned {prune_worst} samples in place: {sample_file}")
        else:
            prune_output = (
                Path(args.prune_output)
                if args.prune_output
                else session_dir / "samples_pruned.json"
            )
            save_json(prune_output, pruned_payload)
            print(f"Wrote pruned sample file to {prune_output}")
        print(
            "Removed sample indices:",
            [sample_error["index"] for sample_error in best_sample_ranking[:prune_worst]],
        )


if __name__ == "__main__":
    main()
