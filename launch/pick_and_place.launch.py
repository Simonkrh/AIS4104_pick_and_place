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
    image_topic = LaunchConfiguration("image_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_realsense", default_value="false"),
            DeclareLaunchArgument("launch_moveit", default_value="true"),
            DeclareLaunchArgument("moveit_launch_rviz", default_value="true"),
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
            DeclareLaunchArgument("model", default_value="models/pick_place_best.pt"),
            DeclareLaunchArgument("conf", default_value="0.4"),
            DeclareLaunchArgument("device", default_value="cpu"),
            DeclareLaunchArgument(
                "handeye_result_file",
                default_value="calibration/eye_in_hand_charuco/handeye_result.json",
            ),
            DeclareLaunchArgument(
                "handeye_child_frame",
                default_value="",
                description=(
                    "Optional TF child frame for the calibrated camera pose, "
                    "for example realsense_cam_link."
                ),
            ),
            DeclareLaunchArgument("pick_approach_offset_z", default_value="0.10"),
            DeclareLaunchArgument("pick_grasp_offset_z", default_value="0.02"),
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
                    {"child_frame": LaunchConfiguration("handeye_child_frame")},
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
                    "launch_rviz": LaunchConfiguration("moveit_launch_rviz")
                }.items(),
            ),
            Node(
                package="depth_localizer",
                executable="pick_moveit_executor_node",
                name="pick_moveit_executor_node",
                output="screen",
            ),
        ]
    )
