import os

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    launch_robot_state_publisher = LaunchConfiguration("launch_robot_state_publisher")
    package_share = Path(get_package_share_directory("ur3e_description"))
    semantic_robot_name = "ur3e_cell"

    custom_urdf_path = str(package_share / "urdf" / "ur3e_cell.urdf.xacro")
    custom_srdf_path = str(package_share / "srdf" / "ur3e_cell.srdf.xacro")
    trac_ik_kinematics_path = str(package_share / "config" / "trac_ik_kinematics.yaml")
    custom_joint_limits_path = str(package_share / "config" / "joint_limits.yaml")

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="ur_moveit_config")
        .robot_description(custom_urdf_path)
        .robot_description_semantic(
            custom_srdf_path,
            {"name": semantic_robot_name},
        )
        .robot_description_kinematics(trac_ik_kinematics_path)
        .joint_limits(custom_joint_limits_path)
        .to_moveit_configs()
    )

    ompl_pipeline = moveit_config.planning_pipelines.get("ompl", {})
    request_adapters = list(ompl_pipeline.get("request_adapters", []))
    ompl_pipeline["request_adapters"] = [
        adapter
        for adapter in request_adapters
        if adapter != "default_planning_request_adapters/CheckStartStateBounds"
    ]

    warehouse_ros_config = {
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "warehouse_host": os.path.expanduser("~/.ros/warehouse_ros.sqlite"),
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            warehouse_ros_config,
            {
                "use_sim_time": False,
                "publish_robot_description_semantic": True,
            },
        ],
    )

    default_rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("ur_moveit_config"), "config", "moveit.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            warehouse_ros_config,
            {
                "use_sim_time": False,
            },
        ],
    )

    robot_state_publisher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                FindPackageShare("ur3e_description"),
                "/launch/ur3e_cell_rsp.launch.py",
            ]
        ),
        condition=IfCondition(launch_robot_state_publisher),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz_config_file,
            ),
            DeclareLaunchArgument(
                "launch_robot_state_publisher",
                default_value="true",
            ),
            robot_state_publisher_launch,
            move_group_node,
            rviz_node,
        ]
    )
