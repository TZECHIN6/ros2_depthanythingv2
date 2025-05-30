import cv2
import numpy as np
import time
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image as PILImage
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "output")
input_image_path = os.path.join(
    script_dir, "input", "captured_image_wheeltec_640x480.jpg"
)
output_depth_path = os.path.join(output_dir, "depth_from_captured_image.npy")

# Create the input directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

save_numpy = True


def save_raw_depth(depth_array, output_depth_path):
    """
    Save the raw depth numpy array for later comparison.
    """
    np.save(output_depth_path, depth_array)
    print(f"Raw depth saved to {output_depth_path}")


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
    max_depth=20.0,
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

# # Apply calibration (replace a and b with your fitted values)
a = 0.845984  # <-- set this to your fitted scale
b = -0.425169  # <-- set this to your fitted offset
depth_np = a * depth_np + b

if save_numpy:
    save_raw_depth(depth_np, output_depth_path)
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
