import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointField, PointCloud2, CompressedImage
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
import numba
from PIL import Image as PILImage
import os
import open3d as o3d
import time
import math

@numba.njit(parallel=True, fastmath=True)
def pack_points_and_colors(x, y, z, colors):
    n = x.size
    points = np.empty((n, 3), np.float32)
    rgbs = np.empty(n, np.uint32)
    for i in numba.prange(n):
        points[i, 0] = x[i] * z[i]
        points[i, 1] = y[i] * z[i]
        points[i, 2] = z[i]
        rgbs[i] = (
            (0xFF << 24)
            | (int(colors[i, 0]) << 16)
            | (int(colors[i, 1]) << 8)
            | int(colors[i, 2])
        )
    return points, rgbs


class DepthToPointCloud(Node):
    def __init__(self):
        super().__init__("depth_to_pointcloud")
        self.declare_parameter("image_transport", "raw")  # raw or compressed (image_transport_py is not available in ROS2 humble yet)
        self.declare_parameter("save_to_ply", False)
        self.declare_parameter("ply_output_dir", "/tmp")
        self.declare_parameter("model_size", "Base")  # Large, Base, Small
        self.declare_parameter("downsample", 3)
        self.declare_parameter("projection_method", "camera_info")  # camera_info, fov
        self.declare_parameter("fovx_deg", 65.0)  # Only used if projection_method is fov

        self.image_transport = self.get_parameter("image_transport").get_parameter_value().string_value
        self.save_to_ply = self.get_parameter("save_to_ply").get_parameter_value().bool_value
        self.ply_output_dir = self.get_parameter("ply_output_dir").get_parameter_value().string_value
        self.model_size = self.get_parameter("model_size").get_parameter_value().string_value
        self.use_sim_time = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        self.downsample = self.get_parameter("downsample").get_parameter_value().integer_value
        self.projection_method = self.get_parameter("projection_method").get_parameter_value().string_value
        self.fovx_deg = self.get_parameter("fovx_deg").get_parameter_value().double_value

        # Validate model_size
        valid_model_sizes = ["Base", "Small", "Large"]
        if self.model_size not in valid_model_sizes:
            self.get_logger().error(
                f"Invalid model_size parameter: '{self.model_size}'. "
                f"Valid options are: {valid_model_sizes}. Node will not start."
            )
            rclpy.shutdown()
            return

        # Validate projection_method
        valid_projection_methods = ["camera_info", "fov"]
        if self.projection_method not in valid_projection_methods:
            self.get_logger().error(
                f"Invalid projection_method parameter: '{self.projection_method}'. "
                f"Valid options are: {valid_projection_methods}. Node will not start."
            )
            rclpy.shutdown()
            return

        # Set up QoS to always use latest image
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        if self.image_transport == "compressed":
            self.image_subscriber = self.create_subscription(
                CompressedImage, "image", self.image_callback, qos_profile
            )
        elif self.image_transport == "raw":
            self.image_subscriber = self.create_subscription(
                Image, "image", self.image_callback, qos_profile
            )
        else:
            self.get_logger().error(
                f"Invalid image_transport parameter: '{self.image_transport}'. "
                "Valid options are: 'raw' or 'compressed'. Node will not start."
            )
            rclpy.shutdown()
            return
        
        self.camera_info_subscriber = self.create_subscription(CameraInfo, "camera_info", self.camera_info_callback, 10)
        self.pointcloud_publisher = self.create_publisher(PointCloud2, "pointcloud", 10)
        if self.use_sim_time:
            self.clock_subscriber = self.create_subscription(Clock, "/clock", self.clock_callback, 10)

        self.get_logger().info(f"Image transport set to: {self.image_transport}")
        self.get_logger().info(f"Subscribed to image topic: {self.resolve_topic_name('image')}")
        self.get_logger().info(f"Subscribed to camera info topic: {self.resolve_topic_name('camera_info')}")
        self.get_logger().info(f"Publishing pointcloud to topic: {self.resolve_topic_name('pointcloud')}")
        self.get_logger().info(f"Projection method set to: {self.projection_method}")

        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Load the model and processor
        model_id = (
            f"depth-anything/Depth-Anything-V2-Metric-Indoor-{self.model_size}-hf"
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            do_resize=True,                      # Enable resizing
            size={"height": 518, "width": 518},  # Target size
            keep_aspect_ratio=True,              # Maintain aspect ratio
            ensure_multiple_of=14,               # Ensure dimensions are multiples of 14
            resample=3,                          # Cubic interpolation (PIL.Image.BICUBIC)
            do_normalize=True,                   # Enable normalization
            image_mean=[0.485, 0.456, 0.406],    # Normalization mean
            image_std=[0.229, 0.224, 0.225],     # Normalization std
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id,
            device_map="auto",
            depth_estimation_type="metric",
            max_depth=20.0,                      # 20 for indoor, 80 for outdoor
        )
        self.get_logger().info(f"Model loaded successfully. Model ID: {model_id}")
        self.get_logger().info(f"Downsample factor set to: {self.downsample}")

        self.bridge = CvBridge()
        self.camera_info = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.image_width = None
        self.image_height = None
        self.image_callback_count = 0
        self.sim_time = None

    def clock_callback(self, msg: Clock):
        if self.use_sim_time:
            self.sim_time = msg.clock
        else:
            self.sim_time = None

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_info is not None:
            return
        self.camera_info = msg
        self.fx = msg.k[0]  # Focal length x
        self.fy = msg.k[4]  # Focal length y
        self.cx = msg.k[2]  # Principal point x
        self.cy = msg.k[5]  # Principal point y
        self.image_width = msg.width
        self.image_height = msg.height
        self.get_logger().info(f"Focal length (fx,fy): ({self.fx},{self.fy})")
        self.get_logger().info(f"Optical center (cx,cy): ({self.cx},{self.cy})")
        self.get_logger().info(f"Image size: {self.image_width}x{self.image_height}")

        # Precompute meshgrid for the expected image size
        self._meshgrid_cache = None  # Reset cache

    def image_callback(self, msg: Image):
        if self.camera_info is None:
            self.get_logger().warn("Camera info not received yet.")
            return

        self.image_callback_count += 1

        # Convert the ROS Image message to a CV2 image
        if self.image_transport == "compressed":
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
        elif self.image_transport == "raw":
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        else:
            self.get_logger().error(
                f"Invalid image_transport parameter: '{self.image_transport}'. "
                "Valid options are: 'raw' or 'compressed'. Node will not process image."
            )
            return

        # Convert the CV2 image to a PIL image
        pil_image = PILImage.fromarray(cv_image)

        # Preprocess the image and predict depth
        t0 = time.time()
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
        depth_np = prediction.squeeze()  # Remove batch dimension
        t1 = time.time()
        inference_time = t1 - t0

        # Apply calibration (replace a and b with your fitted values)
        a = 0.700360
        b = -0.498511
        depth_np = a * depth_np + b
        # Mask invalid (non-positive) depths
        depth_np[depth_np <= 0] = np.nan

        # Create point cloud
        t2 = time.time()
        pointcloud: PointCloud2 = self.create_pointcloud(depth_np, cv_image)
        t3 = time.time()
        pointcloud_time = t3 - t2

        self.get_logger().info(
            f"Inference: {inference_time:.3f}s, PointCloud: {pointcloud_time:.3f}s, Points: {pointcloud.width}"
        )

        # Publish point cloud
        self.pointcloud_publisher.publish(pointcloud)

    def create_pointcloud(self, depth_np, color_image):
        z = np.squeeze(depth_np)
        ds = self.downsample
        z_ds = z[::ds, ::ds]
        color_ds = color_image[::ds, ::ds, :]
        h_ds, w_ds = z_ds.shape

        # Precompute meshgrid only once for each image size and projection method
        cache_key = (self.projection_method, h_ds, w_ds, self.fovx_deg if self.projection_method == "fov" else None)
        if (
            not hasattr(self, "_meshgrid_cache")
            or self._meshgrid_cache is None
            or self._meshgrid_cache.get("key") != cache_key
        ):
            self.get_logger().info("Creating new meshgrid cache for point cloud projection.")
            if self.projection_method == "fov":
                AR = self.image_width / self.image_height
                # For downsampled grid:
                x_idx = np.arange(w_ds)
                y_idx = np.arange(h_ds)
                x_norm = x_idx / (w_ds - 1)
                y_norm = y_idx / (h_ds - 1)
                x_norm_grid, y_norm_grid = np.meshgrid(x_norm, y_norm)
                # FOV projection
                fovx = math.radians(self.fovx_deg)
                x = (np.tan(0.5 * fovx) * (x_norm_grid - 0.5) * 2).astype(np.float32)
                y = (-np.tan(0.5 * fovx / AR) * (0.5 - y_norm_grid) * 2).astype(np.float32)
            elif self.projection_method == "camera_info":
                x_idx = np.arange(w_ds, dtype=np.float32)
                y_idx = np.arange(h_ds, dtype=np.float32)
                x = (x_idx[None, :] - self.cx / ds) / (self.fx / ds)
                y = (y_idx[:, None] - self.cy / ds) / (self.fy / ds)
                x = np.tile(x, (h_ds, 1))
                y = np.tile(y, (1, w_ds))
            self._meshgrid_cache = {
                "key": cache_key,
                "x": x,
                "y": y,
            }
        x = self._meshgrid_cache["x"]
        y = self._meshgrid_cache["y"]

        # Flatten arrays
        z_flat = z_ds.ravel()
        x_flat = x.ravel()
        y_flat = y.ravel()
        colors_flat = color_ds.reshape(-1, 3)

        # Use numba to pack points and colors
        points, rgbs = pack_points_and_colors(x_flat, y_flat, z_flat, colors_flat)
        points_with_color = np.empty((points.shape[0], 4), np.float32)
        points_with_color[:, :3] = points
        points_with_color[:, 3] = rgbs.view(np.float32)

        # Save to PLY if needed
        if self.image_callback_count == 10 and self.save_to_ply:
            colors_ply = colors_flat / 255.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_with_color[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(colors_ply)
            ply_path = os.path.join(
                self.ply_output_dir,
                f"pointcloud_{self.image_callback_count}.ply",
            )
            if not os.path.isdir(self.ply_output_dir):
                self.get_logger().warn(
                    f"PLY output directory does not exist: {self.ply_output_dir}. "
                    "Please create it to enable PLY saving."
                )
            else:
                o3d.io.write_point_cloud(ply_path, pcd)
                self.get_logger().info(f"Saved pointcloud to PLY file: {ply_path}")

        # Create PointCloud2 message
        pointcloud_msg = PointCloud2()
        if self.use_sim_time and self.sim_time is not None:
            pointcloud_msg.header.stamp = self.sim_time
        else:
            pointcloud_msg.header.stamp = self.get_clock().now().to_msg()
        pointcloud_msg.header.frame_id = self.camera_info.header.frame_id
        pointcloud_msg.height = 1
        pointcloud_msg.width = points_with_color.shape[0]
        pointcloud_msg.is_dense = False
        pointcloud_msg.is_bigendian = False
        pointcloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]
        pointcloud_msg.point_step = 16
        pointcloud_msg.row_step = pointcloud_msg.point_step * points_with_color.shape[0]
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
