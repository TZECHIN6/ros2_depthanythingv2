import cv2
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "input")
input_image_path = os.path.join(input_dir, "captured_image_wheeltec_640x480.jpg")

# Create the input directory if it doesn't exist
os.makedirs(input_dir, exist_ok=True)

# Initialize the USB camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Set camera resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

time.sleep(2)  # Allow the camera to warm up

# Capture and discard a few frames to let exposure settle
for _ in range(5):
    cap.read()

# Capture a single frame
ret, frame = cap.read()

# Check if the frame was captured successfully
if ret:
    # Save the captured image
    cv2.imwrite(input_image_path, frame)
    print(f"Image captured and saved at {input_image_path}")
else:
    print("Error: Could not capture image.")

# Release the camera
cap.release()
