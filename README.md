# AIS4104 Pick and Place

This workspace is set up to consume a RealSense feed published from another machine by default.

Set your workspace path once:

```bash
export WORKSPACE=~/path/to/AIS4104_pick_and_place
```

## 1) Remote Camera Requirements

On the camera machine, publish these topics into the same ROS 2 graph:

- `/realsense_cam/color/image_raw`
- `/realsense_cam/aligned_depth_to_color/image_raw`
- `/realsense_cam/color/camera_info`

Make sure both machines share the same `ROS_DOMAIN_ID`, and that `ROS_LOCALHOST_ONLY` is unset or `0`.

If you only publish RGB and not depth, launch this project with `use_depth_localizer:=false`.

## 2) Install Workspace Dependencies

Install required tools/packages and resolve ROS dependencies:

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip
sudo rosdep init   # run once per machine (ignore if already initialized)
rosdep update
cd "$WORKSPACE"
rosdep install --from-paths src --ignore-src -r -y
pip install ultralytics opencv-python
```

You do not need `librealsense`, `realsense-ros`, or `realsense2_camera` on this machine unless you want to plug the camera in locally.

## 3) Build Workspace

```bash
cd "$WORKSPACE"
colcon build --symlink-install
```

## 4) Source Environment

Use this in each new terminal before running:

```bash
source /opt/ros/jazzy/setup.bash
cd "$WORKSPACE"
source install/setup.bash
```

## 5) Default Launch

The default launch now assumes the camera is remote:

```bash
ros2 launch ./launch/pick_and_place.launch.py
```

## 6) Common Launch Modes

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

## 7) Optional Local RealSense Setup

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
