from pathlib import Path

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _is_enabled(value: str) -> bool:
    return value.strip().lower() == "true"


def _maybe_launch_realsense(context):
    if not _is_enabled(LaunchConfiguration("use_realsense").perform(context)):
        return []

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [FindPackageShare("realsense2_camera"), "/launch/rs_launch.py"]
            ),
            launch_arguments={
                "camera_namespace": "",
                "camera_name": "realsense_cam",
                "align_depth.enable": "true",
                "align.enable": "true",
                "pointcloud.enable": "false",
                "depth_module.profile": "640x480x30",
                "rgb_camera.profile": "640x480x15",
            }.items(),
        )
    ]


def generate_launch_description():
    repo_root = Path(__file__).resolve().parents[1]
    default_model_path = str(repo_root / "models" / "pick_place_best.pt")
    default_handeye_result_path = str(
        repo_root / "calibration" / "eye_in_hand_charuco" / "handeye_result.json"
    )

    image_topic = LaunchConfiguration("image_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_realsense", default_value="false"),
            DeclareLaunchArgument("launch_moveit", default_value="true"),
            DeclareLaunchArgument("moveit_launch_rviz", default_value="true"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=[
                    FindPackageShare("ur_moveit_config"),
                    "/config/moveit.rviz",
                ],
            ),
            DeclareLaunchArgument(
                "image_topic", default_value="/realsense_cam/color/image_raw"
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/realsense_cam/aligned_depth_to_color/image_raw",
            ),
            DeclareLaunchArgument(
                "camera_info_topic", default_value="/realsense_cam/color/camera_info"
            ),
            DeclareLaunchArgument("model", default_value=default_model_path),
            DeclareLaunchArgument("conf", default_value="0.4"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument(
                "handeye_result_file",
                default_value=default_handeye_result_path,
            ),
            DeclareLaunchArgument("pick_approach_offset_z", default_value="0.1"),
            DeclareLaunchArgument("pick_grasp_offset_z", default_value="-0.02"),
            DeclareLaunchArgument(
                "pick_approach_fallback_enabled", default_value="true"
            ),
            DeclareLaunchArgument(
                "pick_approach_fallback_xy_step", default_value="0.02"
            ),
            DeclareLaunchArgument(
                "pick_approach_fallback_xy_levels", default_value="2"
            ),
            DeclareLaunchArgument(
                "pick_approach_fallback_z_step", default_value="0.01"
            ),
            DeclareLaunchArgument("pick_approach_fallback_z_levels", default_value="2"),
            DeclareLaunchArgument("pick_pre_grasp_clearance_z", default_value="0.05"),
            DeclareLaunchArgument("pick_grasp_velocity_scaling", default_value="0.15"),
            DeclareLaunchArgument(
                "pick_grasp_acceleration_scaling", default_value="0.05"
            ),
            DeclareLaunchArgument("pick_table_top_z", default_value="-0.015"),
            DeclareLaunchArgument("pick_min_grasp_clearance_z", default_value="0.01"),
            DeclareLaunchArgument(
                "pick_tool_yaw",
                default_value="3.14159265359",
                description=(
                    "Tool yaw for picking. pi points the gripper the usual way here."
                ),
            ),
            DeclareLaunchArgument(
                "pick_approach_camera_offset_y",
                default_value="0.10",
                description=(
                    "Camera-to-gripper Y offset for centering above the object."
                ),
            ),
            DeclareLaunchArgument("robot_ip", default_value="192.168.0.100"),
            DeclareLaunchArgument(
                "ready_joint_positions_deg",
                default_value="[-90.0, -90.0, 0.0, -180.0, 90.0, 180.0]",
                description="Ready pose, in UR joint degrees.",
            ),
            DeclareLaunchArgument(
                "search_start_joint_positions_deg",
                default_value="[-80.0, -105.0, 0.0, -163.0, 90.0, 190.0]",
                description="First pose for looking over the workspace.",
            ),
            DeclareLaunchArgument(
                "search_joint_positions_deg",
                default_value="[]",
                description=(
                    "Extra full search poses, if the small look-around is not enough."
                ),
            ),
            DeclareLaunchArgument(
                "search_look_offsets_deg",
                default_value=(
                    "[[0.0, 0.0, 0.0, 5.0, 10.0, 0.0], "
                    "[0.0, 0.0, 0.0, 5.0, -8.0, 0.0], "
                    "[0.0, 0.0, 0.0, 10.0, 0.0, 0.0], "
                    "[0.0, -9.0, 0.0, -17.0, -3.0, 8.0], "
                    "[0.0, 0.0, 0.0, -19.0, 20.0, -6.0]]"
                ),
                description=(
                    "Small offsets from the search-start pose for pointing the camera."
                ),
            ),
            DeclareLaunchArgument("search_pose_wait_sec", default_value="1.0"),
            OpaqueFunction(function=_maybe_launch_realsense),
            Node(
                package="yolo_detector",
                executable="yolo_node",
                name="yolo_node",
                output="screen",
                parameters=[
                    {"image_topic": image_topic},
                    {"model": LaunchConfiguration("model")},
                    {"conf": LaunchConfiguration("conf")},
                    {"device": LaunchConfiguration("device")},
                ],
            ),
            Node(
                package="opencv_processor",
                executable="opencv_edges_node",
                name="opencv_edges_node",
                output="screen",
                parameters=[{"image_topic": image_topic}],
            ),
            Node(
                package="depth_localizer",
                executable="detection_3d_node",
                name="detection_3d_node",
                output="screen",
                parameters=[
                    {"depth_topic": depth_topic},
                    {"camera_info_topic": camera_info_topic},
                ],
            ),
            Node(
                package="depth_localizer",
                executable="handeye_static_tf_publisher",
                name="handeye_static_tf_publisher",
                output="screen",
                parameters=[
                    {"handeye_result_file": LaunchConfiguration("handeye_result_file")},
                    {"child_frame": "realsense_cam_link"},
                ],
            ),
            Node(
                package="depth_localizer",
                executable="detection_3d_transform_node",
                name="detection_3d_transform_node",
                output="screen",
            ),
            Node(
                package="depth_localizer",
                executable="pick_pose_generator_node",
                name="pick_pose_generator_node",
                output="screen",
                parameters=[
                    {
                        "approach_offset_z": LaunchConfiguration(
                            "pick_approach_offset_z"
                        )
                    },
                    {"grasp_offset_z": LaunchConfiguration("pick_grasp_offset_z")},
                    {"table_top_z": LaunchConfiguration("pick_table_top_z")},
                    {
                        "min_grasp_clearance_z": LaunchConfiguration(
                            "pick_min_grasp_clearance_z"
                        )
                    },
                    {"tool_yaw": LaunchConfiguration("pick_tool_yaw")},
                    {
                        "approach_camera_offset_y": LaunchConfiguration(
                            "pick_approach_camera_offset_y"
                        )
                    },
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        FindPackageShare("ur3e_description"),
                        "/launch/ur3e_cell_moveit.launch.py",
                    ]
                ),
                condition=IfCondition(LaunchConfiguration("launch_moveit")),
                launch_arguments={
                    "launch_rviz": LaunchConfiguration("moveit_launch_rviz"),
                    "rviz_config": LaunchConfiguration("rviz_config"),
                }.items(),
            ),
            Node(
                package="depth_localizer",
                executable="pick_moveit_executor_node",
                name="pick_moveit_executor_node",
                output="screen",
                parameters=[
                    {
                        "ready_joint_positions_deg": LaunchConfiguration(
                            "ready_joint_positions_deg"
                        )
                    },
                    {
                        "search_start_joint_positions_deg": LaunchConfiguration(
                            "search_start_joint_positions_deg"
                        )
                    },
                    {
                        "search_joint_positions_deg": ParameterValue(
                            LaunchConfiguration("search_joint_positions_deg"),
                            value_type=str,
                        )
                    },
                    {
                        "search_look_offsets_deg": ParameterValue(
                            LaunchConfiguration("search_look_offsets_deg"),
                            value_type=str,
                        )
                    },
                    {
                        "search_pose_wait_sec": LaunchConfiguration(
                            "search_pose_wait_sec"
                        )
                    },
                    {"robot_ip": LaunchConfiguration("robot_ip")},
                    {
                        "approach_fallback_enabled": LaunchConfiguration(
                            "pick_approach_fallback_enabled"
                        )
                    },
                    {
                        "approach_fallback_xy_step": LaunchConfiguration(
                            "pick_approach_fallback_xy_step"
                        )
                    },
                    {
                        "approach_fallback_xy_levels": LaunchConfiguration(
                            "pick_approach_fallback_xy_levels"
                        )
                    },
                    {
                        "approach_fallback_z_step": LaunchConfiguration(
                            "pick_approach_fallback_z_step"
                        )
                    },
                    {
                        "approach_fallback_z_levels": LaunchConfiguration(
                            "pick_approach_fallback_z_levels"
                        )
                    },
                    {
                        "pre_grasp_clearance_z": LaunchConfiguration(
                            "pick_pre_grasp_clearance_z"
                        )
                    },
                    {
                        "grasp_velocity_scaling": LaunchConfiguration(
                            "pick_grasp_velocity_scaling"
                        )
                    },
                    {
                        "grasp_acceleration_scaling": LaunchConfiguration(
                            "pick_grasp_acceleration_scaling"
                        )
                    },
                ],
            ),
        ]
    )
