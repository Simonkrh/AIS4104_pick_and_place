# AIS4104 Pick and Place

## RealSense driver

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

## ROS Wrapper

cd ~/GitHubRepos/AIS4104_pick_and_place/src
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros
git checkout 4.0.4

## Build Workspace

cd ~/GitHubRepos/AIS4104_pick_and_place
colcon build --symlink-install --packages-up-to realsense2_camera
colcon build --symlink-install

## Run

source /opt/ros/humble/setup.bash
source ~/GitHubRepos/AIS4104_pick_and_place/install/setup.bash
ros2 launch ./launch/pick_and_place.launch.py

## Launch

source /opt/ros/humble/setup.bash
source ~/GitHubRepos/AIS4104_pick_and_place/install/setup.bash
ros2 launch ./launch/pick_and_place.launch.py

### YOLO/OpenCV on USB image, RealSense for depth/pointcloud (default behavior)

ros2 launch ./launch/pick_and_place.launch.py \
 use_realsense:=true \
 use_usb_cam:=true \
 image_topic:=/usb_cam/image_raw

### RealSense only (YOLO/OpenCV use RealSense color image)

source /opt/ros/humble/setup.bash
source ~/GitHubRepos/AIS4104_pick_and_place/install/setup.bash
ros2 launch ./launch/pick_and_place.launch.py \
 use_realsense:=true \
 use_usb_cam:=false \
 image_topic:=/camera/color/image_raw

### USB camera only (no RealSense node)

source /opt/ros/humble/setup.bash
source ~/GitHubRepos/AIS4104_pick_and_place/install/setup.bash
ros2 launch ./launch/pick_and_place.launch.py \
 use_realsense:=false \
 use_usb_cam:=true \
 image_topic:=/usb_cam/image_raw

### Pointcloud ON with true RGB texture (real color)

source /opt/ros/humble/setup.bash
source ~/GitHubRepos/AIS4104_pick_and_place/install/setup.bash
ros2 launch ./launch/pick_and_place.launch.py \
 use_realsense:=true \
 pointcloud_enable:=true \
 align_depth_enable:=true \
 pointcloud_stream_filter:=2 \
 pointcloud_stream_index_filter:=0

### Pointcloud ON with "Any" texture stream (more robust / fewer texture warnings)

source /opt/ros/humble/setup.bash
source ~/GitHubRepos/AIS4104_pick_and_place/install/setup.bash
ros2 launch ./launch/pick_and_place.launch.py \
 use_realsense:=true \
 pointcloud_enable:=true \
 align_depth_enable:=true \
 pointcloud_stream_filter:=0 \
 pointcloud_stream_index_filter:=0
