# AIS4104 Pick and Place

This workspace is set up to consume a RealSense feed published from another machine by default.

## Prerequisites

- Ubuntu 24.04 LTS
- ROS 2 Jazzy installed on this machine
- A remote machine publishing the RealSense topics into the same ROS 2 network
- Matching `ROS_DOMAIN_ID` on both machines
- `ROS_LOCALHOST_ONLY=0` or unset on both machines

The project has been tested with ROS 2 Jazzy on Ubuntu 24.04. A good starting point is the official ROS 2 Jazzy Ubuntu install guide:

https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

`ros-jazzy-ros-base` is sufficient for this project. You do not need the full desktop install unless you also want extra GUI tools.

## Installation

### 1) Install ROS 2 Jazzy

Follow the official ROS 2 Jazzy install instructions for Ubuntu 24.04, then verify that this works:

```bash
source /opt/ros/jazzy/setup.bash
ros2 --help
```

### 2) Clone the workspace

Choose a workspace location and clone this repository:

```bash
mkdir -p ~/ais4104_ws/src
cd ~/ais4104_ws/src
git clone https://github.com/Simonkrh/AIS4104_pick_and_place.git AIS4104_pick_and_place
```

Set your workspace path:

```bash
export WORKSPACE=~/ais4104_ws
```

### 3) Install workspace tools and dependencies

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip
sudo rosdep init   # run once per machine (ignore if already initialized)
rosdep update
source /opt/ros/jazzy/setup.bash
cd "$WORKSPACE"
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install --user --break-system-packages -r src/AIS4104_pick_and_place/requirements.txt
```

You do not need `librealsense`, `realsense-ros`, or `realsense2_camera` on this machine unless you want to plug the camera in locally.

### 4) Build the workspace

```bash
source /opt/ros/jazzy/setup.bash
cd "$WORKSPACE"
colcon build --symlink-install
```

### 5) Source the environment

Use this in each new terminal before running:

```bash
source /opt/ros/jazzy/setup.bash
cd "$WORKSPACE"
source install/setup.bash
```

## Remote Camera Requirements

On the camera machine, publish these topics into the same ROS 2 graph:

- `/realsense_cam/color/image_raw`
- `/realsense_cam/aligned_depth_to_color/image_raw`
- `/realsense_cam/color/camera_info`

Make sure both machines share the same `ROS_DOMAIN_ID`, and that `ROS_LOCALHOST_ONLY` is unset or `0`.

If you only publish RGB and not depth, launch this project with `use_depth_localizer:=false`.

## Default Launch

The default launch now assumes the camera is remote:

```bash
ros2 launch ./launch/pick_and_place.launch.py
```

### Test the gripper manually

The launch defaults to `robot_ip:=192.168.0.100`. If the robot IP changes, pass `robot_ip:=<ip>` when launching.

```bash
ros2 service call /pick_moveit_executor_node/open_gripper std_srvs/srv/Trigger {}
ros2 service call /pick_moveit_executor_node/close_gripper std_srvs/srv/Trigger {}
```

## Common Launch Modes

### Use a specific model

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  model:=models/pick_place_best.pt
```

### Remote camera with custom topic names

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  image_topic:=/my_camera/color/image_raw \
  depth_topic:=/my_camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/my_camera/color/camera_info
```

### Remote RGB only

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  use_depth_localizer:=false
```

## Optional Local RealSense Setup

Only do this if you want the RealSense physically attached to this machine.

### Install RealSense driver (`librealsense`)

```bash
cd ~
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.50.0
rm -rf build
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_EXAMPLES=true \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j"$(nproc)"
sudo make install
sudo ldconfig
sudo ../scripts/setup_udev_rules.sh
```

Then unplug and replug the RealSense camera.

### Add ROS wrapper (`realsense-ros`)

```bash
cd "$WORKSPACE/src"
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros
git checkout 4.0.4
```

### Build the camera wrapper

```bash
cd "$WORKSPACE"
colcon build --symlink-install --packages-up-to realsense2_camera
colcon build --symlink-install
```

### Launch with the local camera

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  use_realsense:=true
```

The pointcloud-related launch arguments are only relevant in this local-camera mode.
