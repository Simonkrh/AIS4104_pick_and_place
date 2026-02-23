from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    use_realsense = LaunchConfiguration("use_realsense")

    image_topic = LaunchConfiguration("image_topic")
    detection_topic = LaunchConfiguration("detection_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    model = LaunchConfiguration("model")
    conf = LaunchConfiguration("conf")
    device = LaunchConfiguration("device")
    use_depth_localizer = LaunchConfiguration("use_depth_localizer")
    align_depth_enable = LaunchConfiguration("align_depth_enable")
    pointcloud_enable = LaunchConfiguration("pointcloud_enable")
    depth_profile = LaunchConfiguration("depth_profile")
    rgb_profile = LaunchConfiguration("rgb_profile")
    pointcloud_stream_filter = LaunchConfiguration("pointcloud_stream_filter")
    pointcloud_stream_index_filter = LaunchConfiguration(
        "pointcloud_stream_index_filter"
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("realsense2_camera"), "/launch/rs_launch.py"]
        ),
        condition=IfCondition(use_realsense),
        launch_arguments={
            "camera_namespace": "",
            "camera_name": "realsense_cam",
            "align_depth.enable": align_depth_enable,
            "pointcloud.enable": pointcloud_enable,
            "depth_module.profile": depth_profile,
            "rgb_camera.profile": rgb_profile,
            "pointcloud.stream_filter": pointcloud_stream_filter,
            "pointcloud.stream_index_filter": pointcloud_stream_index_filter,
        }.items(),
    )

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
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_realsense", default_value="true"),
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
            DeclareLaunchArgument("align_depth_enable", default_value="true"),
            DeclareLaunchArgument("pointcloud_enable", default_value="true"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("rgb_profile", default_value="1280x720x30"),
            DeclareLaunchArgument("pointcloud_stream_filter", default_value="0"),
            DeclareLaunchArgument("pointcloud_stream_index_filter", default_value="0"),
            DeclareLaunchArgument("conf", default_value="0.4"),
            DeclareLaunchArgument("device", default_value="cpu"),
            realsense_launch,
            yolo,
            opencv,
            depth_localizer,
        ]
    )
