import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointField, PointCloud2
from cv_bridge import CvBridge

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image as PILImage

import os
import open3d as o3d


class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__("depth_to_pointcloud")
        self.declare_parameter("image_topic", "/logi_camera/image_raw")
        self.declare_parameter("camera_info_topic", "/logi_camera/camera_info")
        self.declare_parameter("pointcloud_topic", "/logi_camera/pointcloud")
        self.declare_parameter("save_to_ply", False)

        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.pointcloud_topic = (
            self.get_parameter("pointcloud_topic").get_parameter_value().string_value
        )
        self.save_to_ply = (
            self.get_parameter("save_to_ply").get_parameter_value().bool_value
        )

        self.image_subscriber = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10
        )
        self.pointcloud_publisher = self.create_publisher(
            PointCloud2, self.pointcloud_topic, 10
        )

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Load the model and processor
        model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            do_resize=True,  # Enable resizing
            size={"height": 518, "width": 518},  # Target size
            keep_aspect_ratio=True,  # Maintain aspect ratio
            ensure_multiple_of=14,  # Ensure dimensions are multiples of 14
            resample=3,  # Cubic interpolation (PIL.Image.BICUBIC)
            do_normalize=True,  # Enable normalization
            image_mean=[0.485, 0.456, 0.406],  # Normalization mean
            image_std=[0.229, 0.224, 0.225],  # Normalization std
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            device_map="auto",
            depth_estimation_type="metric",
            max_depth=20.0,
        )

        self.bridge = CvBridge()
        self.camera_info = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.image_callback_count = 0

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info is not None:
            return

        # Store the camera info message
        self.camera_info = msg

        # Extract focal lengths (fx, fy) from the camera intrinsic matrix K
        self.fx = msg.k[0]  # Focal length x
        self.fy = msg.k[4]  # Focal length y

        # Extract optical centers (cx, cy)
        self.cx = msg.k[2]  # Principal point x
        self.cy = msg.k[5]  # Principal point y

        self.get_logger().info(f"Focal length (fx,fy): ({self.fx},{self.fy})")
        self.get_logger().info(f"Optical center (cx,cy): ({self.cx},{self.cy})")

    def image_callback(self, msg: Image):
        if self.camera_info is None:
            self.get_logger().warn("Camera info not received yet.")
            return

        self.image_callback_count += 1

        # Convert the ROS Image message to a CV2 image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        # Convert the CV2 image to a PIL image
        pil_image = PILImage.fromarray(cv_image)

        # Preprocess the image and predict depth
        inputs = self.image_processor(
            pil_image,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(pil_image.height, pil_image.width),
            mode="bilinear",
            align_corners=True,
        )
        prediction = prediction.cpu().numpy()

        # Create point cloud
        pointcloud: PointCloud2 = self.create_pointcloud(
            pil_image.height, pil_image.width, prediction, cv_image
        )

        # Publish point cloud
        self.pointcloud_publisher.publish(pointcloud)
        self.get_logger().info(
            f"Published point cloud with {pointcloud.width} points to {self.pointcloud_topic}",
            throttle_duration_sec=1,
        )

    def create_pointcloud(self, h, w, depth_np, color_image):
        # Create a meshgrid of pixel coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Calculate coordinates using the center-based approach
        x = (x - w / 2) / self.fx
        y = (y - h / 2) / self.fy
        z = np.array(depth_np)

        # Stack and reshape points with colors
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(
            -1, 3
        )
        colors = np.array(color_image).reshape(-1, 3)

        # Create the point cloud and save it to the output directory
        if self.image_callback_count in (1, 10, 20) and self.save_to_ply:
            colors_ply = np.array(color_image).reshape(-1, 3) / 255.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors_ply)
            o3d.io.write_point_cloud(
                os.path.join(
                    f"/home/thomasluk/Code/ros2_ws/pointcloud_{self.image_callback_count}.ply",
                ),
                pcd,
            )

        # Create PointCloud2 message
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header.stamp = self.get_clock().now().to_msg()
        pointcloud_msg.header.frame_id = self.camera_info.header.frame_id
        pointcloud_msg.height = 1
        pointcloud_msg.width = points.shape[0]
        pointcloud_msg.is_dense = True
        pointcloud_msg.is_bigendian = False

        # Define fields for XYZ and RGB
        pointcloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        # Point step is now 16 bytes (12 for XYZ + 4 for RGB)
        pointcloud_msg.point_step = 16
        pointcloud_msg.row_step = pointcloud_msg.point_step * points.shape[0]

        # Combine XYZ and RGB into a single array
        rgb = colors.astype(np.uint32)
        # Pack RGB into RGBA (alpha = 255)
        rgb_packed = np.array(
            (255 << 24) | (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | (rgb[:, 2]),
            dtype=np.uint32,
        )
        points_with_color = np.zeros((points.shape[0], 4), dtype=np.float32)
        points_with_color[:, 0:3] = points
        points_with_color[:, 3] = rgb_packed.view(np.float32)

        pointcloud_msg.data = points_with_color.tobytes()

        return pointcloud_msg


def main(args=None):
    rclpy.init(args=args)
    depth_to_pointcloud = DepthToPointCloud()
    rclpy.spin(depth_to_pointcloud)
    depth_to_pointcloud.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
