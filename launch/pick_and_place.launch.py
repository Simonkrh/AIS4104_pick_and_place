from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    use_realsense = LaunchConfiguration("use_realsense")
    use_usb_cam = LaunchConfiguration("use_usb_cam")

    video_device = LaunchConfiguration("video_device")
    pixel_format = LaunchConfiguration("pixel_format")
    image_width = LaunchConfiguration("image_width")
    image_height = LaunchConfiguration("image_height")
    framerate = LaunchConfiguration("framerate")

    image_topic = LaunchConfiguration("image_topic")
    conf = LaunchConfiguration("conf")
    device = LaunchConfiguration("device")
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
            "camera_name": "camera",
            "align_depth.enable": align_depth_enable,
            "pointcloud.enable": pointcloud_enable,
            "depth_module.profile": depth_profile,
            "rgb_camera.profile": rgb_profile,
            "pointcloud.stream_filter": pointcloud_stream_filter,
            "pointcloud.stream_index_filter": pointcloud_stream_index_filter,
        }.items(),
    )

    usb_cam = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="usb_cam",
        output="screen",
        condition=IfCondition(use_usb_cam),
        parameters=[
            {"video_device": video_device},
            {"pixel_format": pixel_format},
            {"image_width": image_width},
            {"image_height": image_height},
            {"framerate": framerate},
        ],
        remappings=[
            ("image_raw", "/usb_cam/image_raw"),
            ("camera_info", "/usb_cam/camera_info"),
        ],
    )

    yolo = Node(
        package="yolo_detector",
        executable="yolo_node",
        name="yolo_node",
        output="screen",
        parameters=[
            {"image_topic": image_topic},
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

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_realsense", default_value="true"),
            DeclareLaunchArgument("use_usb_cam", default_value="true"),
            DeclareLaunchArgument("video_device", default_value="/dev/video0"),
            DeclareLaunchArgument("pixel_format", default_value="mjpeg2rgb"),
            DeclareLaunchArgument("image_width", default_value="1920"),
            DeclareLaunchArgument("image_height", default_value="1080"),
            DeclareLaunchArgument("framerate", default_value="30.0"),
            # Keep perception on USB camera by default; RealSense is used for depth/pointcloud
            DeclareLaunchArgument("image_topic", default_value="/usb_cam/image_raw"),
            DeclareLaunchArgument("align_depth_enable", default_value="true"),
            DeclareLaunchArgument("pointcloud_enable", default_value="true"),
            DeclareLaunchArgument("depth_profile", default_value="640x480x30"),
            DeclareLaunchArgument("rgb_profile", default_value="1280x720x30"),
            # Use "Any" texture stream by default to avoid repeated "No stream match ... Color" warnings
            DeclareLaunchArgument("pointcloud_stream_filter", default_value="0"),
            DeclareLaunchArgument("pointcloud_stream_index_filter", default_value="0"),
            DeclareLaunchArgument("conf", default_value="0.4"),
            DeclareLaunchArgument("device", default_value="cpu"),
            realsense_launch,
            usb_cam,
            yolo,
            opencv,
        ]
    )
