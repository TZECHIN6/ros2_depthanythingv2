import os
import numpy as np
import matplotlib.pyplot as plt

input_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
depth_1_file_name = "frame0000.npy"
depth_2_file_name = "frame0001.npy"
depth_1_dir = os.path.join(input_data_dir, depth_1_file_name)
depth_2_dir = os.path.join(input_data_dir, depth_2_file_name)

depth1 = np.load(depth_1_dir)
depth2 = np.load(depth_2_dir)
diff = np.abs(depth1 - depth2)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title(f"{depth_1_file_name}")
plt.imshow(depth1, cmap="plasma")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title(f"{depth_2_file_name}")
plt.imshow(depth2, cmap="plasma")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Absolute Difference")
plt.imshow(diff, cmap="viridis")
plt.colorbar()

plt.tight_layout()
plt.show()
