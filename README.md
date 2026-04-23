# AIS4104 Pick and Place

This project runs a UR3e pick-and-place pipeline with:

- YOLO object detection
- RealSense RGB/depth input
- 3D object localization
- eye-in-hand calibration transform
- MoveIt motion planning
- OnRobot 2FG7 gripper commands

The normal setup is that the RealSense camera topics already exist on the ROS 2 network. The launch file can also start a local RealSense camera if needed.

## Prerequisites

- Ubuntu 24.04 LTS
- ROS 2 Jazzy
- UR / MoveIt packages installed through `rosdep`
- Matching `ROS_DOMAIN_ID` on all machines
- `ROS_LOCALHOST_ONLY=0` or unset on all machines

The project has been tested with ROS 2 Jazzy on Ubuntu 24.04.

ROS 2 install guide:

https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

## Installation

### 1) Clone the workspace

```bash
mkdir -p ~/ais4104_ws/src
cd ~/ais4104_ws/src
git clone https://github.com/Simonkrh/AIS4104_pick_and_place.git AIS4104_pick_and_place
```

Set your workspace path:

```bash
export WORKSPACE=~/ais4104_ws
```

### 2) Install dependencies

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip
sudo rosdep init   # run once per machine, ignore if already done
rosdep update

source /opt/ros/jazzy/setup.bash
cd "$WORKSPACE"
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install --user --break-system-packages -r src/AIS4104_pick_and_place/requirements.txt
```

### 3) Build

```bash
source /opt/ros/jazzy/setup.bash
cd "$WORKSPACE"
colcon build --symlink-install
```

### 4) Source in every new terminal

```bash
source /opt/ros/jazzy/setup.bash
cd "$WORKSPACE"
source install/setup.bash
cd src/AIS4104_pick_and_place
```

## Camera Topics

By default, the project expects these topics:

- `/realsense_cam/color/image_raw`
- `/realsense_cam/aligned_depth_to_color/image_raw`
- `/realsense_cam/color/camera_info`

If the camera runs on another machine, make sure both machines are on the same ROS 2 network and use the same `ROS_DOMAIN_ID`.

## Launch

### Robot description only

Use this if you only want the UR3e cell robot description and TF:

```bash
ros2 launch ur3e_description ur3e_cell_rsp.launch.py
```

### Full pick-and-place launch

Run this from the repository folder:

```bash
ros2 launch ./launch/pick_and_place.launch.py
```

Use GPU device `0` for YOLO:

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  device:='"0"'
```

Use a custom RViz file:

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  device:='"0"' \
  rviz_config:=/path/to/your/file.rviz
```

Disable RViz:

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  moveit_launch_rviz:=false
```

Use another robot IP:

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  robot_ip:=192.168.0.100
```

Use other camera topics:

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  image_topic:=/my_camera/color/image_raw \
  depth_topic:=/my_camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/my_camera/color/camera_info
```

Start a local RealSense camera from this launch:

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  use_realsense:=true
```

## Useful Services

Move robot to the ready/start pose:

```bash
ros2 service call /pick_moveit_executor_node/move_to_start_pose std_srvs/srv/Trigger "{}"
```

Open and close the gripper:

```bash
ros2 service call /pick_moveit_executor_node/open_gripper std_srvs/srv/Trigger "{}"
ros2 service call /pick_moveit_executor_node/close_gripper std_srvs/srv/Trigger "{}"
```

Move to the current approach pose:

```bash
ros2 service call /pick_moveit_executor_node/execute_approach std_srvs/srv/Trigger "{}"
```

Move to the current grasp pose:

```bash
ros2 service call /pick_moveit_executor_node/execute_grasp std_srvs/srv/Trigger "{}"
```

Run open gripper, approach, grasp, and close gripper as one command:

```bash
ros2 service call /pick_moveit_executor_node/execute_pick std_srvs/srv/Trigger "{}"
```

There is also a small helper script for the ready/start pose:

```bash
./tools/move_to_start_pose.sh
```

## Calibration

The default calibration folder is:

```bash
calibration/eye_in_hand_charuco
```

The default result file used by the launch is:

```bash
calibration/eye_in_hand_charuco/handeye_result.json
```

### 1) Generate the ChArUco board

```bash
python3 tools/generate_charuco_board.py
```

Print the generated board at 100% scale.

### 2) Collect samples

Start the robot/camera launch first, then run:

```bash
python3 tools/collect_eye_in_hand_samples.py
```

Move the robot to different stable poses. Press `s` to save a sample and `q` to quit. Aim for about 15-30 good samples.

Optional automatic capture from hardcoded good poses:

```bash
python3 tools/auto_collect_eye_in_hand_samples.py
```

Preview without moving:

```bash
python3 tools/auto_collect_eye_in_hand_samples.py --dry-run
```

Automatic samples are saved in `calibration/eye_in_hand_auto`.

### 3) Solve calibration

```bash
python3 tools/solve_eye_in_hand.py
```

Solve automatic samples:

```bash
python3 tools/solve_eye_in_hand.py --session-dir calibration/eye_in_hand_auto
```

If you use another calibration file, pass it to the launch:

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  handeye_result_file:=/path/to/handeye_result.json
```

## Important Launch Arguments

- `device`: YOLO device. Use `cpu` or `'"0"'`.
- `model`: YOLO model path. Default is `models/pick_place_best.pt`.
- `conf`: YOLO confidence threshold. Default is `0.4`.
- `robot_ip`: robot IP for gripper URScript commands. Default is `192.168.0.100`.
- `rviz_config`: RViz file to load.
- `handeye_result_file`: hand-eye calibration result JSON.
- `pick_approach_offset_z`: approach height above target. Default is `0.1`.
- `pick_grasp_offset_z`: grasp height above target. Default is `-0.02`.
- `pick_tool_yaw`: fixed tool yaw in radians. Default is `3.14159265359`, which flips the camera/gripper to face away from the robot.
- `pick_approach_camera_offset_x`: camera X offset from `gripper_tcp` in meters, used only to shift the approach pose. Default is `0.0`.
- `pick_approach_camera_offset_y`: camera Y offset from `gripper_tcp` in meters, used only to shift the approach pose so the camera is above the target. Default is `0.10`.
- `ready_joint_positions_deg`: ready pose joint angles in degrees.
