from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def _maybe_launch_realsense(context):
    use_realsense = LaunchConfiguration("use_realsense").perform(context).lower()
    if use_realsense not in ("1", "true", "yes", "on"):
        return []

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [FindPackageShare("realsense2_camera"), "/launch/rs_launch.py"]
            ),
            launch_arguments={
                "camera_namespace": "",
                "camera_name": "realsense_cam",
                "align_depth.enable": LaunchConfiguration("align_depth_enable"),
                "align.enable": LaunchConfiguration("align_depth_enable"),
                "pointcloud.enable": LaunchConfiguration("pointcloud_enable"),
                "depth_module.profile": LaunchConfiguration("depth_profile"),
                "rgb_camera.profile": LaunchConfiguration("rgb_profile"),
                "pointcloud.stream_filter": LaunchConfiguration(
                    "pointcloud_stream_filter"
                ),
                "pointcloud.stream_index_filter": LaunchConfiguration(
                    "pointcloud_stream_index_filter"
                ),
            }.items(),
        )
    ]


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    detection_topic = LaunchConfiguration("detection_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    model = LaunchConfiguration("model")
    conf = LaunchConfiguration("conf")
    device = LaunchConfiguration("device")
    use_depth_localizer = LaunchConfiguration("use_depth_localizer")
    detections_3d_topic = LaunchConfiguration("detections_3d_topic")

    yolo = Node(
        package="yolo_detector",
        executable="yolo_node",
        name="yolo_node",
        output="screen",
        parameters=[
            {"image_topic": image_topic},
            {"model": model},
            {"conf": conf},
            {"device": device},
        ],
    )

    opencv = Node(
        package="opencv_processor",
        executable="opencv_edges_node",
        name="opencv_edges_node",
        output="screen",
        parameters=[{"image_topic": image_topic}],
    )

    depth_localizer = Node(
        package="depth_localizer",
        executable="detection_3d_node",
        name="detection_3d_node",
        output="screen",
        condition=IfCondition(use_depth_localizer),
        parameters=[
            {"detection_topic": detection_topic},
            {"depth_topic": depth_topic},
            {"camera_info_topic": camera_info_topic},
            {"output_topic": detections_3d_topic},
        ],
    )

    handeye_tf = Node(
        package="depth_localizer",
        executable="handeye_static_tf_publisher",
        name="handeye_static_tf_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("publish_handeye_tf")),
        parameters=[
            {"handeye_result_file": LaunchConfiguration("handeye_result_file")},
            {"parent_frame": LaunchConfiguration("handeye_parent_frame")},
            {"child_frame": LaunchConfiguration("handeye_child_frame")},
        ],
    )

    detections_tf = Node(
        package="depth_localizer",
        executable="detection_3d_transform_node",
        name="detection_3d_transform_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("publish_transformed_detections")),
        parameters=[
            {"input_topic": detections_3d_topic},
            {"output_topic": LaunchConfiguration("detections_3d_base_topic")},
            {"target_frame": LaunchConfiguration("detection_target_frame")},
            {"tf_timeout_sec": LaunchConfiguration("tf_timeout_sec")},
            {"allow_latest_tf_fallback": LaunchConfiguration("allow_latest_tf_fallback")},
            {"best_pose_topic": LaunchConfiguration("best_pose_topic")},
            {"best_tf_child_frame": LaunchConfiguration("best_tf_child_frame")},
            {"target_class": LaunchConfiguration("target_class")},
            {"min_score": LaunchConfiguration("min_score")},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_realsense", default_value="false"),
            DeclareLaunchArgument("use_depth_localizer", default_value="true"),
            DeclareLaunchArgument(
                "image_topic", default_value="/realsense_cam/color/image_raw"
            ),
            DeclareLaunchArgument("detection_topic", default_value="/yolo/detections"),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/realsense_cam/aligned_depth_to_color/image_raw",
            ),
            DeclareLaunchArgument(
                "camera_info_topic", default_value="/realsense_cam/color/camera_info"
            ),
            DeclareLaunchArgument("model", default_value="models/pick_place_best.pt"),
            DeclareLaunchArgument(
                "detections_3d_topic", default_value="/yolo/detections_3d"
            ),
            DeclareLaunchArgument(
                "publish_handeye_tf",
                default_value="true",
                description="Publish the hand-eye tool->camera static TF from handeye_result.json.",
            ),
            DeclareLaunchArgument(
                "handeye_result_file",
                default_value="calibration/eye_in_hand_charuco/handeye_result.json",
            ),
            DeclareLaunchArgument(
                "handeye_parent_frame",
                default_value="",
                description="Override parent frame for hand-eye TF (defaults to metadata.tool_frame).",
            ),
            DeclareLaunchArgument(
                "handeye_child_frame",
                default_value="",
                description="Override child frame for hand-eye TF (defaults to metadata.camera.image_frame).",
            ),
            DeclareLaunchArgument(
                "publish_transformed_detections",
                default_value="true",
                description="Transform Detection3DArray into the robot frame using TF.",
            ),
            DeclareLaunchArgument(
                "detections_3d_base_topic", default_value="/yolo/detections_3d_base"
            ),
            DeclareLaunchArgument("detection_target_frame", default_value="base"),
            DeclareLaunchArgument("tf_timeout_sec", default_value="0.05"),
            DeclareLaunchArgument("allow_latest_tf_fallback", default_value="true"),
            DeclareLaunchArgument("target_class", default_value=""),
            DeclareLaunchArgument("min_score", default_value="0.0"),
            DeclareLaunchArgument("best_pose_topic", default_value="/pick_target_pose"),
            DeclareLaunchArgument("best_tf_child_frame", default_value="detected_object"),
            DeclareLaunchArgument("align_depth_enable", default_value="true"),
            DeclareLaunchArgument("pointcloud_enable", default_value="true"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("rgb_profile", default_value="1280x720x30"),
            DeclareLaunchArgument("pointcloud_stream_filter", default_value="0"),
            DeclareLaunchArgument("pointcloud_stream_index_filter", default_value="0"),
            DeclareLaunchArgument("conf", default_value="0.4"),
            DeclareLaunchArgument("device", default_value="cpu"),
            OpaqueFunction(function=_maybe_launch_realsense),
            yolo,
            opencv,
            depth_localizer,
            handeye_tf,
            detections_tf,
        ]
    )
