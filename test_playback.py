# Standard Library
import os
import math
import json

# External Library
import numpy as np

# Internal Library
from test_teaching import calculate_T_base_gripper, calculate_T_cam_template


def transform_to_pose6dof_deg(T: np.ndarray) -> np.ndarray:
    """
    !Convert a 4x4 homogeneous transform to [x, y, z, rx, ry, rz]
    @T (np.ndarray): input 4x4 transform matrix
    """
    if T.shape != (4, 4):
        raise ValueError("Input must be a 4x4 homogeneous matrix")

    R = T[:3, :3]
    t = T[:3, 3]

    # Ensure R is a proper rotation (orthonormalize if necessary)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Extract Euler angles (roll-pitch-yaw, X-Y-Z) in radians
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0

    # Convert radians → degrees
    rx, ry, rz = map(math.degrees, [roll, pitch, yaw])

    # Output [x, y, z, rx, ry, rz]
    pose_6dof = np.array([t[0] * 1000, t[1] * 1000, t[2] * 1000, rx, ry, rz])
    return pose_6dof


def compute_safe_top_down_pose(
    T: np.ndarray, desired_tool_z=(0, 0, -1.0)
) -> np.ndarray:
    """
    Forces the pose so that the tool Z axis points in desired_tool_z direction
    (default: downward in base frame), keeps the same position, and keeps yaw
    by projecting the original X axis into the horizontal plane.
    """

    T = np.array(T, dtype=float)
    R = T[:3, :3]
    pos = T[:3, 3]

    # normalize desired Z
    z_new = np.array(desired_tool_z, dtype=float)
    z_new = z_new / np.linalg.norm(z_new)

    # Project original X axis onto plane orthogonal to new Z
    x_old = R[:, 0]
    x_proj = x_old - np.dot(x_old, z_new) * z_new

    # Handle degenerate case (x_proj too small)
    if np.linalg.norm(x_proj) < 1e-6:
        # choose arbitrary X direction not parallel to Z
        if abs(z_new[2]) < 0.9:
            x_proj = np.array([0, 0, 1], dtype=float)
        else:
            x_proj = np.array([1, 0, 0], dtype=float)
        x_proj = x_proj - np.dot(x_proj, z_new) * z_new

    x_new = x_proj / np.linalg.norm(x_proj)
    y_new = np.cross(z_new, x_new)
    y_new = y_new / np.linalg.norm(y_new)

    # rebuild rotation
    R_new = np.column_stack((x_new, y_new, z_new))

    # build T_safe
    T_safe = np.eye(4)
    T_safe[:3, :3] = R_new
    T_safe[:3, 3] = pos

    return T_safe


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    T_base_gripper = calculate_T_base_gripper()
    hand_eye_path = os.path.join(current_dir, "data", "hand_eye_params.json")
    with open(hand_eye_path, "r") as f:
        hand_eye_params = json.load(f)
    T_gripper_cam = np.array(hand_eye_params["T_gripper_cam"])
    T_cam_template = calculate_T_cam_template(
        "target_23.jpg", "template_edge.png", "plane_2.jpg"
    )
    T_base_template = T_base_gripper @ T_gripper_cam @ T_cam_template

    # Load collected points
    points_path = os.path.join(current_dir, "data", "collected_points.json")
    with open(points_path, "r") as f:
        collected_points = json.load(f)
    for point in collected_points:
        T_template_cam = np.array(point["T_template_target"])
        T_base_target = T_base_template @ T_template_cam
        target_pose = transform_to_pose6dof_deg(T_base_target)
        target_pose = compute_safe_top_down_pose(T_base_target)
        print(f"Point ID {point['id']} - Target Pose [mm, deg]: {target_pose}")
