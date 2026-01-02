import numpy as np
import json
import os

# Define paths
calibration_folder = os.path.join(
    os.path.dirname(__file__), "..", "calibration"
)
matrix_path = os.path.join(calibration_folder, "calibration_matrix_usb.npy")
dist_path = os.path.join(calibration_folder, "distortion_coefficients_usb.npy")
output_path = os.path.join(calibration_folder, "camera_params.json")

# Load numpy arrays
K = np.load(matrix_path)
D = np.load(dist_path)

# Create dictionary with camera parameters
camera_params = {"K": K.tolist(), "D": D.tolist()}

# Save to JSON file
with open(output_path, "w") as f:
    json.dump(camera_params, f, indent=4)

print(f"Camera parameters saved to: {output_path}")
print(f"K shape: {K.shape}")
print(f"D shape: {D.shape}")
