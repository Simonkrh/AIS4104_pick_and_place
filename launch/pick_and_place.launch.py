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
    sync_queue_size = LaunchConfiguration("sync_queue_size")
    sync_slop = LaunchConfiguration("sync_slop")
    use_pick_pose_generator = LaunchConfiguration("use_pick_pose_generator")
    use_moveit_executor = LaunchConfiguration("use_moveit_executor")

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
            {"sync_queue_size": sync_queue_size},
            {"sync_slop": sync_slop},
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

    pick_pose_generator = Node(
        package="depth_localizer",
        executable="pick_pose_generator_node",
        name="pick_pose_generator_node",
        output="screen",
        condition=IfCondition(use_pick_pose_generator),
        parameters=[
            {"input_topic": LaunchConfiguration("best_pose_topic")},
            {"approach_topic": LaunchConfiguration("pick_approach_topic")},
            {"grasp_topic": LaunchConfiguration("pick_grasp_topic")},
            {"approach_offset_z": LaunchConfiguration("pick_approach_offset_z")},
            {"grasp_offset_z": LaunchConfiguration("pick_grasp_offset_z")},
            {"tool_roll": LaunchConfiguration("pick_tool_roll")},
            {"tool_pitch": LaunchConfiguration("pick_tool_pitch")},
            {"tool_yaw": LaunchConfiguration("pick_tool_yaw")},
            {"publish_tf": LaunchConfiguration("pick_pose_publish_tf")},
            {"approach_tf_child_frame": LaunchConfiguration("pick_approach_tf_child_frame")},
            {"grasp_tf_child_frame": LaunchConfiguration("pick_grasp_tf_child_frame")},
        ],
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ur3e_description"), "/launch/ur3e_cell_moveit.launch.py"]
        ),
        condition=IfCondition(LaunchConfiguration("launch_moveit")),
        launch_arguments={
            "launch_rviz": LaunchConfiguration("moveit_launch_rviz"),
        }.items(),
    )

    pick_moveit_executor = Node(
        package="depth_localizer",
        executable="pick_moveit_executor_node",
        name="pick_moveit_executor_node",
        output="screen",
        condition=IfCondition(use_moveit_executor),
        parameters=[
            {"approach_topic": LaunchConfiguration("pick_approach_topic")},
            {"grasp_topic": LaunchConfiguration("pick_grasp_topic")},
            {"status_topic": LaunchConfiguration("pick_execution_status_topic")},
            {"ur_type": LaunchConfiguration("ur_type")},
            {"group_name": LaunchConfiguration("moveit_group_name")},
            {"base_link_name": LaunchConfiguration("moveit_base_link")},
            {"end_effector_name": LaunchConfiguration("moveit_end_effector_link")},
            {"target_link": LaunchConfiguration("moveit_target_link")},
            {"max_pose_age_sec": LaunchConfiguration("pick_pose_max_age_sec")},
            {"position_tolerance": LaunchConfiguration("moveit_position_tolerance")},
            {"orientation_tolerance": LaunchConfiguration("moveit_orientation_tolerance")},
            {"cartesian_grasp": LaunchConfiguration("moveit_cartesian_grasp")},
            {"cartesian_max_step": LaunchConfiguration("moveit_cartesian_max_step")},
            {"wait_for_moveit_seconds": LaunchConfiguration("wait_for_moveit_seconds")},
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
            DeclareLaunchArgument("use_pick_pose_generator", default_value="true"),
            DeclareLaunchArgument(
                "use_moveit_executor",
                default_value="true",
                description="Enable the MoveIt-based service node for approach/grasp execution.",
            ),
            DeclareLaunchArgument(
                "launch_moveit",
                default_value="false",
                description="Also launch the local UR3e cell MoveIt/move_group on this machine.",
            ),
            DeclareLaunchArgument(
                "moveit_launch_rviz",
                default_value="true",
                description="Launch RViz together with the local MoveIt instance when launch_moveit is true.",
            ),
            DeclareLaunchArgument("ur_type", default_value="ur3e"),
            DeclareLaunchArgument(
                "sync_queue_size",
                default_value="10",
                description="ApproximateTimeSynchronizer queue size for detection_3d_node.",
            ),
            DeclareLaunchArgument(
                "sync_slop",
                default_value="0.10",
                description="ApproximateTimeSynchronizer slop (sec) for detection_3d_node.",
            ),
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
            DeclareLaunchArgument("detection_target_frame", default_value="base_link"),
            DeclareLaunchArgument("tf_timeout_sec", default_value="0.05"),
            DeclareLaunchArgument("allow_latest_tf_fallback", default_value="true"),
            DeclareLaunchArgument("target_class", default_value=""),
            DeclareLaunchArgument("min_score", default_value="0.0"),
            DeclareLaunchArgument("best_pose_topic", default_value="/pick_target_pose"),
            DeclareLaunchArgument("best_tf_child_frame", default_value="detected_object"),
            DeclareLaunchArgument("pick_approach_topic", default_value="/pick_approach_pose"),
            DeclareLaunchArgument("pick_grasp_topic", default_value="/pick_grasp_pose"),
            DeclareLaunchArgument("pick_approach_offset_z", default_value="0.10"),
            DeclareLaunchArgument("pick_grasp_offset_z", default_value="0.02"),
            DeclareLaunchArgument("pick_tool_roll", default_value="3.141592653589793"),
            DeclareLaunchArgument("pick_tool_pitch", default_value="0.0"),
            DeclareLaunchArgument("pick_tool_yaw", default_value="0.0"),
            DeclareLaunchArgument("pick_pose_publish_tf", default_value="true"),
            DeclareLaunchArgument(
                "pick_execution_status_topic", default_value="/pick_execution_status"
            ),
            DeclareLaunchArgument("pick_pose_max_age_sec", default_value="2.0"),
            DeclareLaunchArgument(
                "pick_approach_tf_child_frame", default_value="pick_approach"
            ),
            DeclareLaunchArgument(
                "pick_grasp_tf_child_frame", default_value="pick_grasp"
            ),
            DeclareLaunchArgument("moveit_group_name", default_value="ur_manipulator"),
            DeclareLaunchArgument("moveit_base_link", default_value="base_link"),
            DeclareLaunchArgument("moveit_end_effector_link", default_value="gripper_tcp"),
            DeclareLaunchArgument("moveit_target_link", default_value="gripper_tcp"),
            DeclareLaunchArgument("moveit_position_tolerance", default_value="0.005"),
            DeclareLaunchArgument("moveit_orientation_tolerance", default_value="0.05"),
            DeclareLaunchArgument("moveit_cartesian_grasp", default_value="false"),
            DeclareLaunchArgument("moveit_cartesian_max_step", default_value="0.0025"),
            DeclareLaunchArgument("wait_for_moveit_seconds", default_value="5.0"),
            DeclareLaunchArgument("align_depth_enable", default_value="true"),
            DeclareLaunchArgument("pointcloud_enable", default_value="false"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("rgb_profile", default_value="640x480x15"),
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
            pick_pose_generator,
            moveit_launch,
            pick_moveit_executor,
        ]
    )
