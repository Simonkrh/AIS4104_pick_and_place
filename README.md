# AIS4104 Pick and Place

## 1) Install RealSense Driver (librealsense)

Build and install `librealsense` from source (version `v2.50.0`):

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

Then unplug/replug the RealSense camera.

## 2) Add ROS Wrapper (realsense-ros)

Clone the wrapper into this workspace and pin to the version used here:

```bash
cd ~/GitHubRepos/AIS4104_pick_and_place/src
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros
git checkout 4.0.4
```

## 3) Install Workspace Dependencies

Install required tools/packages and resolve ROS dependencies:

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip
sudo rosdep init   # run once per machine (ignore if already initialized)
rosdep update
cd ~/GitHubRepos/AIS4104_pick_and_place
rosdep install --from-paths src --ignore-src -r -y
pip install ultralytics opencv-python
```

## 4) Build Workspace

```bash
cd ~/GitHubRepos/AIS4104_pick_and_place
colcon build --symlink-install --packages-up-to realsense2_camera
colcon build --symlink-install
```

## 5) Source Environment

Use this in each new terminal before running:

```bash
source /opt/ros/humble/setup.bash
cd <your_workspace>
source install/setup.bash
```

## 6) Default Launch

```bash
ros2 launch ./launch/pick_and_place.launch.py
```

## 7) Common Launch Modes

### Pointcloud ON with true RGB texture (real color)

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  pointcloud_enable:=true \
  align_depth_enable:=true \
  pointcloud_stream_filter:=2 \
  pointcloud_stream_index_filter:=0
```

### Pointcloud ON with "Any" texture stream (more robust / fewer texture warnings)

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  pointcloud_enable:=true \
  align_depth_enable:=true \
  pointcloud_stream_filter:=0 \
  pointcloud_stream_index_filter:=0
```
