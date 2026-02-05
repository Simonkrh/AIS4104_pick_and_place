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
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-pip ros-humble-usb-cam
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

## 5) Configure Stable Sony Camera Alias (`/dev/sony_camera`)

Run once per machine so `video_device:=/dev/sony_camera` is stable.
Note: this rule matches the camera used in this project (`idVendor=0bda`, `idProduct=5805`). If another camera is used, replace those values with the device IDs:

```bash
lsusb
```

Then create the udev rule:

```bash
sudo tee /etc/udev/rules.d/99-sony-camera.rules >/dev/null <<'EOF2'
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="5805", ATTR{index}=="0", SYMLINK+="sony_camera"
EOF2

sudo udevadm control --reload-rules
sudo udevadm trigger
```

Then unplug/replug the Sony camera and verify:

```bash
ls -l /dev/sony_camera
```

## 6) Source Environment

Use this in each new terminal before running:

```bash
source /opt/ros/humble/setup.bash
cd <your_workspace>
source install/setup.bash
```

## 7) Default Launch

```bash
ros2 launch ./launch/pick_and_place.launch.py
```

## 8) Common Launch Modes

### YOLO/OpenCV on Sony image, RealSense for depth/pointcloud (default behavior)

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  use_realsense:=true \
  use_sony_cam:=true \
  image_topic:=/sony_cam/image_raw
```

### RealSense only (YOLO/OpenCV use RealSense color image)

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  use_realsense:=true \
  use_sony_cam:=false \
  image_topic:=/realsense_cam/color/image_raw
```

### Sony camera only (no RealSense node)

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  use_realsense:=false \
  use_sony_cam:=true \
  image_topic:=/sony_cam/image_raw
```

### Pointcloud ON with true RGB texture (real color)

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  use_realsense:=true \
  pointcloud_enable:=true \
  align_depth_enable:=true \
  pointcloud_stream_filter:=2 \
  pointcloud_stream_index_filter:=0
```

### Pointcloud ON with "Any" texture stream (more robust / fewer texture warnings)

```bash
ros2 launch ./launch/pick_and_place.launch.py \
  use_realsense:=true \
  pointcloud_enable:=true \
  align_depth_enable:=true \
  pointcloud_stream_filter:=0 \
  pointcloud_stream_index_filter:=0
```
