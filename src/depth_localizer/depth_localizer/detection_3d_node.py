#!/usr/bin/env python3
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3D, Detection3DArray, Detection2DArray


class Detection3DNode(Node):
    def __init__(self):
        super().__init__("detection_3d_node")

        self.declare_parameter("detection_topic", "/yolo/detections")
        self.declare_parameter(
            "depth_topic", "/realsense_cam/aligned_depth_to_color/image_raw"
        )
        self.declare_parameter("camera_info_topic", "/realsense_cam/color/camera_info")
        self.declare_parameter("output_topic", "/yolo/detections_3d")
        self.declare_parameter("depth_scale", 0.001)  # mm -> m for 16UC1 streams
        self.declare_parameter("roi_half_size", 2)  # 2 => 5x5 patch
        self.declare_parameter("min_depth_m", 0.10)
        self.declare_parameter("max_depth_m", 2.00)
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.10)

        detection_topic = self.get_parameter("detection_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        output_topic = self.get_parameter("output_topic").value

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.roi_half_size = int(self.get_parameter("roi_half_size").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        sync_queue_size = int(self.get_parameter("sync_queue_size").value)
        sync_slop = float(self.get_parameter("sync_slop").value)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Detection3DArray, output_topic, 10)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_frame_id = ""

        self.create_subscription(CameraInfo, camera_info_topic, self.on_camera_info, 10)

        self.det_sub = message_filters.Subscriber(self, Detection2DArray, detection_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.det_sub, self.depth_sub],
            queue_size=sync_queue_size,
            slop=sync_slop,
            allow_headerless=False,
        )
        self.sync.registerCallback(self.on_synced)

        self.get_logger().info(f"Subscribing detections: {detection_topic}")
        self.get_logger().info(f"Subscribing depth: {depth_topic}")
        self.get_logger().info(f"Subscribing camera info: {camera_info_topic}")
        self.get_logger().info(f"Publishing 3D detections: {output_topic}")

    def on_camera_info(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self.camera_frame_id = msg.header.frame_id

    def on_synced(self, det_msg: Detection2DArray, depth_msg: Image):
        if self.fx is None or self.fy is None:
            return

        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        height, width = depth_image.shape[:2]

        out = Detection3DArray()
        out.header = det_msg.header
        out.header.frame_id = depth_msg.header.frame_id or self.camera_frame_id

        converted = 0
        for det in det_msg.detections:
            u = int(round(det.bbox.center.position.x))
            v = int(round(det.bbox.center.position.y))

            u0 = max(0, u - self.roi_half_size)
            u1 = min(width, u + self.roi_half_size + 1)
            v0 = max(0, v - self.roi_half_size)
            v1 = min(height, v + self.roi_half_size + 1)
            if u0 >= u1 or v0 >= v1:
                continue

            patch = depth_image[v0:v1, u0:u1]
            depth_values_m = self.depth_patch_to_meters(patch, depth_msg.encoding)
            if depth_values_m.size == 0:
                continue

            z = float(np.median(depth_values_m))
            if not (self.min_depth_m <= z <= self.max_depth_m):
                continue

            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy

            det3d = Detection3D()
            det3d.header = out.header
            det3d.results = det.results

            # 3D center estimate in camera optical frame.
            det3d.bbox.center.position.x = x
            det3d.bbox.center.position.y = y
            det3d.bbox.center.position.z = z
            det3d.bbox.center.orientation.w = 1.0

            det3d.bbox.size.x = max((det.bbox.size_x * z) / self.fx, 0.0)
            det3d.bbox.size.y = max((det.bbox.size_y * z) / self.fy, 0.0)
            det3d.bbox.size.z = 0.05

            for hyp in det3d.results:
                hyp.pose.pose.position.x = x
                hyp.pose.pose.position.y = y
                hyp.pose.pose.position.z = z
                hyp.pose.pose.orientation.w = 1.0

            out.detections.append(det3d)
            converted += 1

        self.pub.publish(out)
        if converted > 0:
            self.get_logger().debug(f"Published {converted} detections with 3D points")

    def depth_patch_to_meters(self, patch: np.ndarray, encoding: str) -> np.ndarray:
        if patch.size == 0:
            return np.array([], dtype=np.float32)

        if encoding in ("16UC1", "mono16"):
            meters = patch.astype(np.float32) * self.depth_scale
        else:
            meters = patch.astype(np.float32)

        valid = np.isfinite(meters) & (meters > 0.0)
        if not np.any(valid):
            return np.array([], dtype=np.float32)
        return meters[valid]


def main():
    rclpy.init()
    node = Detection3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
