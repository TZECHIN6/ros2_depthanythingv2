import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
input_image_path = os.path.join(script_dir, "input", "captured_image.jpg")

# Initialize the USB camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture a single frame
ret, frame = cap.read()

# Check if the frame was captured successfully
if ret:
    # Save the captured image
    cv2.imwrite(input_image_path, frame)
    print("Image captured and saved.")
else:
    print("Error: Could not capture image.")

# Release the camera
cap.release()