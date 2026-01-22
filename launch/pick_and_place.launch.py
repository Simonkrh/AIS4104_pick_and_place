from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    video_device = LaunchConfiguration("video_device")
    pixel_format = LaunchConfiguration("pixel_format")
    image_width = LaunchConfiguration("image_width")
    image_height = LaunchConfiguration("image_height")
    framerate = LaunchConfiguration("framerate")

    image_topic = LaunchConfiguration("image_topic")
    conf = LaunchConfiguration("conf")
    device = LaunchConfiguration("device")

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("realsense2_camera"), "/launch/rs_launch.py"]
        ),
        launch_arguments={
            "align_depth.enable": "true",
            "pointcloud.enable": "true",
            "pointcloud.texture_stream": "RS2_STREAM_DEPTH",
        }.items(),
    )

    usb_cam = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="usb_cam",
        output="screen",
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
        parameters=[
            {"image_topic": image_topic},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("video_device", default_value="/dev/video0"),
            DeclareLaunchArgument("pixel_format", default_value="mjpeg2rgb"),
            DeclareLaunchArgument("image_width", default_value="1920"),
            DeclareLaunchArgument("image_height", default_value="1080"),
            DeclareLaunchArgument("framerate", default_value="30.0"),
            DeclareLaunchArgument("image_topic", default_value="/usb_cam/image_raw"),
            DeclareLaunchArgument("conf", default_value="0.4"),
            DeclareLaunchArgument("device", default_value="cpu"),
            realsense_launch,
            usb_cam,
            yolo,
            opencv,
        ]
    )
