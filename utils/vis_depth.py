import cv2
import numpy as np
import time
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image as PILImage
import os
import open3d as o3d
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output")
input_image_path = os.path.join(script_dir, "input", "frame0002.jpg")
output_depth_path = os.path.join(output_dir, "frame0002.npy")

# Create the input directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

focal_length_x = 605.00671
focal_length_y = 607.04704
cx = 268.86013
cy = 239.53292
save_numpy = True
save_ply = True


def save_raw_depth(depth_array, output_depth_path):
    """
    Save the raw depth numpy array for later comparison.
    """
    np.save(output_depth_path, depth_array)
    print(f"Raw depth saved to {output_depth_path}")


def save_ply_file(depth_array, rgb_image, output_dir, filename):
    """
    Save the depth map as a PLY file.
    """
    # Generate mesh grid and calculate point cloud coordinates
    width = depth_array.shape[1]
    height = depth_array.shape[0]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - cx) / focal_length_x
    y = (y - cy) / focal_length_y
    z = np.squeeze(depth_array)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(rgb_image).reshape(-1, 3) / 255.0

    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    out_file = os.path.join(
            output_dir, filename + "_camera_info.ply",
        )
    o3d.io.write_point_cloud(
        out_file,
        pcd,
    )
    print(f"PLY file saved to {out_file}")


def save_ply_file_fov(depth_array, rgb_image, output_dir, filename, fovx_deg=65):
    """
    Save the depth map as a PLY file using FOV-based projection.
    """
    height, width = depth_array.shape
    AR = width / height
    fovx = math.radians(fovx_deg)
    # Normalize pixel coordinates to [0, 1]
    x_idx = np.arange(width)
    y_idx = np.arange(height)
    x_norm = x_idx / (width - 1)
    y_norm = y_idx / (height - 1)
    x_norm_grid, y_norm_grid = np.meshgrid(x_norm, y_norm)

    D = depth_array  # Already calibrated

    # Compute 3D coordinates
    x_3d = D * math.tan(0.5 * fovx) * (x_norm_grid - 0.5) * 2
    y_3d = -D * math.tan(0.5 * fovx / AR) * (0.5 - y_norm_grid) * 2
    z_3d = D

    points = np.stack((x_3d, y_3d, z_3d), axis=-1).reshape(-1, 3)
    colors = np.array(rgb_image).reshape(-1, 3) / 255.0

    # Remove points where D is nan or <= 0
    valid = np.isfinite(z_3d).ravel() & (z_3d.ravel() > 0)
    points = points[valid]
    colors = colors[valid]

    # Save as PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    out_file = os.path.join(
        output_dir, filename + "_fov.ply"
    )
    o3d.io.write_point_cloud(
        out_file,
        pcd,
    )
    print(f"PLY file (FOV projection) saved to {out_file}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
model_size = "Large"
model_id = f"depth-anything/Depth-Anything-V2-Metric-Indoor-{model_size}-hf"
image_processor = AutoImageProcessor.from_pretrained(
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
model = AutoModelForDepthEstimation.from_pretrained(
    model_id,
    device_map="auto",
    depth_estimation_type="metric",
    max_depth=1.0,
)
print(f"Model loaded successfully. Model ID: {model_id}")

# Load the image from file
cv_image_bgr = cv2.imread(input_image_path)
cv_image = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2RGB)
pil_image = PILImage.fromarray(cv_image)

# Preprocess the image and predict depth
t0 = time.time()
inputs = image_processor(
    pil_image,
    return_tensors="pt",
)
inputs = inputs.to(device)
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# interpolate to original size
prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=(pil_image.height, pil_image.width),
    mode="bilinear",
    align_corners=True,
)
prediction = prediction.cpu().numpy()
t1 = time.time()
inference_time = t1 - t0

# Generate depth map
depth_np = prediction.squeeze()  # Remove batch dimension

# Apply calibration (replace a and b with your fitted values)
a = 12.597215
b = -0.238963
new_depth_np = a * depth_np + b

if save_numpy:
    save_raw_depth(depth_np, output_depth_path)
if save_ply:
    filename = os.path.splitext(os.path.basename(input_image_path))[0]
    filename1 = filename + "_raw_depth"
    filename2 = filename + "_cali_depth"
    print(filename1, filename2)
    save_ply_file(depth_np, pil_image, output_dir, filename1)
    save_ply_file(new_depth_np, pil_image, output_dir, filename2)
    save_ply_file_fov(depth_np, pil_image, output_dir, filename1, fovx_deg=65)
    save_ply_file_fov(new_depth_np, pil_image, output_dir, filename2, fovx_deg=65)

depth_display = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Mouse callback to show depth value at pointer and record on click
current_depth = None
current_pos = None

clicked_points = []  # To store (predicted_depth, real_depth, (u, v))


def mouse_callback(event, u, v, flags, param):
    global current_depth, current_pos
    if event == cv2.EVENT_MOUSEMOVE:
        if 0 <= v < depth_np.shape[0] and 0 <= u < depth_np.shape[1]:
            current_depth = depth_np[v, u]
            current_pos = (u, v)
    elif event == cv2.EVENT_LBUTTONDOWN:
        if 0 <= v < depth_np.shape[0] and 0 <= u < depth_np.shape[1]:
            predicted = depth_np[v, u]
            print(f"Clicked at ({u}, {v}), predicted depth: {predicted:.4f}")
            try:
                real = float(input(f"Enter real depth (meters) for point ({u}, {v}): "))
                clicked_points.append((predicted, real, (u, v)))
                print(f"Saved: predicted={predicted:.4f}, real={real}, at ({u}, {v})")
            except Exception as e:
                print(f"Invalid input: {e}")


cv2.namedWindow("Depth Map")
cv2.setMouseCallback("Depth Map", mouse_callback)

while True:
    display_img = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)
    if current_pos is not None and current_depth is not None:
        text = f"({current_pos[0]}, {current_pos[1]}): z={current_depth:.3f}"
        cv2.putText(
            display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.circle(display_img, current_pos, 3, (0, 255, 0), -1)
    cv2.imshow("Depth Map", display_img)
    cv2.imshow("RGB Image", cv_image_bgr)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()

# Perform least squares fitting if enough points were clicked
if len(clicked_points) >= 2:
    predicted = np.array([pt[0] for pt in clicked_points])
    real = np.array([pt[1] for pt in clicked_points])
    # Fit: real = a * predicted + b
    a, b = np.polyfit(predicted, real, 1)
    print(f"\nCalibration result (least squares fit):")
    print(f"  real_depth = {a:.6f} * predicted_depth + {b:.6f}")
    print(f"  (scale: {a:.6f}, offset: {b:.6f})")
    # Optionally, save calibration data
    # np.save(
    #     os.path.join(output_dir, "calibration_points.npy"),
    #     np.array(clicked_points, dtype=object),
    # )
    # with open(os.path.join(output_dir, "calibration_fit.txt"), "w") as f:
    #     f.write(f"real_depth = {a:.6f} * predicted_depth + {b:.6f}\n")
    # print(f"Calibration points and fit saved to {output_dir}")
else:
    print("Not enough calibration points for fitting (need at least 2).")
