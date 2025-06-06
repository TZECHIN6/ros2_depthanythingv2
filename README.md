# ros2_depthanythingv2

## Overview

`ros2_depthanythingv2` is a ROS 2 package that converts monocular RGB images into depth maps and generates colored 3D point clouds. It leverages DepthAnythingV2 models for depth estimation and publishes the resulting point clouds as ROS 2 messages, suitable for navigation, mapping, and perception tasks.

## Requirements

- ROS 2 (tested on Humble)
- Ubuntu 22.04

## Features

- Subscribes to RGB image and camera info topics
- Runs DepthAnythingV2 model to estimate depth from images
- Projects depth and color into a 3D point cloud using camera intrinsics or FOV equation
- Publishes `sensor_msgs/PointCloud2` messages
- Optionally saves point clouds as PLY files for offline analysis

## Launch

To start the node with default parameters:
```bash
ros2 launch ros2_depthanythingv2 depth_to_pointcloud.launch.py
```

You can customize parameters such as image topic, model size, downsampling factor, and more via launch arguments.

## Parameters

- `image_transport`: Image transport type (e.g., raw, compressed)
- `model_size`: Model variant to use (Base, Small, Large)
- `downsample`: Downsampling factor for point cloud density
- `projection_method`: Use camera_info or FOV equation for projection
- `fovx_deg`: Horizontal field of view (if using FOV projection)
- `max_depth`: Maximum depth value for scaling
- `use_sim_time`: Use simulation time (for simulation)

## Topics

- **Subscribed**
  - `/image_raw` or `/image_compressed` (`sensor_msgs/Image` or `sensor_msgs/CompressedImage`): Input RGB image
  - `/camera_info` (sensor_msgs/CameraInfo): Camera calibration

- **Published**
  - `/pointcloud` (sensor_msgs/PointCloud2): Output colored point cloud

## Example

```bash
ros2 launch ros2_depthanythingv2 depth_to_pointcloud.launch.py image_transport:=compressed model_size:=Base downsample:=3
```

## Usage

This package is originally designed for use with the [Nav2 STVL (Spatio-Temporal Voxel Layer) plugin](https://docs.nav2.org/tutorials/docs/navigation2_with_stvl.html) to mimic a depth camera. By converting monocular RGB images into depth maps and projecting them into colored 3D point clouds, it enables robots without dedicated depth sensors to provide point cloud data for navigation, obstacle avoidance, and mapping tasks in Nav2.

Typical workflow:
1. Launch this package to publish a point cloud from your monocular camera.
2. Configure the Nav2 STVL plugin (or other point cloud consumers) to subscribe to the `/pointcloud` topic published by this node.
3. Tune parameters such as `downsample`, `max_depth`, and `projection_method` for your environment and performance needs.

This approach allows you to leverage advanced depth estimation models to enhance robot perception using only a standard RGB camera.

## Acknowledgements

Special thanks to the authors and contributors of the following open source projects, which made this package possible:

- [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [spatio_temporal_voxel_layer](https://github.com/SteveMacenski/spatio_temporal_voxel_layer)

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

