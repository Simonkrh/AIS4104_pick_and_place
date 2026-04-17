from setuptools import find_packages, setup

package_name = "depth_localizer"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="maintainer",
    maintainer_email="maintainer@example.com",
    description="Fuse 2D detections with aligned depth to estimate 3D positions.",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "detection_3d_node = depth_localizer.detection_3d_node:main",
            "detection_3d_transform_node = depth_localizer.detection_3d_transform_node:main",
            "handeye_static_tf_publisher = depth_localizer.handeye_static_tf_publisher:main",
            "pick_pose_generator_node = depth_localizer.pick_pose_generator_node:main",
            "pick_moveit_executor_node = depth_localizer.pick_moveit_executor_node:main",
        ],
    },
)
