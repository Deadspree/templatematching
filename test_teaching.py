# Standard Library
import os
import math
import json

# External Library
import cv2
import numpy as np
from fairino import Robot

# internal Library
from scripts.invariant_edge_matching import template_matching_pyramid
from scripts.templatepose import (
    new_charuco_pose_estimation,
    get_plane_from_pose,
    perform_pixel_to_plane,
    get_rotation_about_normal,
    convert_to_homogeneous,
)


def load_image(target_image, template_image, plane_image):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_image_path = os.path.join(current_dir, "input", target_image)
    template_image_path = os.path.join(current_dir, "input", template_image)
    plane_image_path = os.path.join(current_dir, "input", plane_image)
    plane = cv2.imread(plane_image_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.medianBlur(target, 5)
    target = cv2.Canny(target, 20, 200)
    template = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
    return target, template, plane


def calculate_T_base_gripper():
    error, pose = get_TCP_pose()
    T_base_gripper = convert_euler_to_homogeneous(
        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
    )
    return T_base_gripper


def get_TCP_pose():
    robot = Robot.RPC("192.168.58.2")
    error, pose = robot.GetActualTCPPose()
    robot.CloseRPC()
    return error, pose


def convert_euler_to_homogeneous(
    x: float, y: float, z: float, rx: float, ry: float, rz: float
) -> np.ndarray:
    """
    !Calculate T_base_ee from the robot pose and save it to transforms.json

    @x (float): x coordinates in mm
    @y (float): y coordinates in mm
    @z (float): z coordinates in mm
    @rx (float): Euler angles degree compared to x axis
    @ry (float): Euler angles degree compared to y axis
    @rz (float): Euler angles degree compared to z axis
    """
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)

    # Rotation matrices
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rx), -math.sin(rx)],
            [0, math.sin(rx), math.cos(rx)],
        ]
    )

    R_y = np.array(
        [
            [math.cos(ry), 0, math.sin(ry)],
            [0, 1, 0],
            [-math.sin(ry), 0, math.cos(ry)],
        ]
    )

    R_z = np.array(
        [
            [math.cos(rz), -math.sin(rz), 0],
            [math.sin(rz), math.cos(rz), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation (Z * Y * X)
    R_ee = R_z @ R_y @ R_x
    scale = 0.001  # for metres conversion
    # Homogeneous transformation
    T_base_ee = np.eye(4)
    T_base_ee[:3, :3] = R_ee
    T_base_ee[:3, 3] = [x * scale, y * scale, z * scale]

    # print("T_base_ee calculated:")
    # print(T_base_ee)

    # JSON_PATH = PROJECT_ROOT / "data" / "transforms.json"

    return T_base_ee


def calculate_T_cam_template(target_image, template_image, plane_image):
    target, template, plane = load_image(
        target_image, template_image, plane_image
    )
    cameraMatrix, distCoeffs = load_camera_calibration()
    squareLength = 0.03
    markerLength = 0.022
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
    board = cv2.aruco.CharucoBoard(
        (9, 6), squareLength, markerLength, dictionary
    )
    _, rvec, tvec = new_charuco_pose_estimation(
        plane, cameraMatrix, distCoeffs, board
    )
    print("rvec:", rvec)
    print("tvec:", tvec)    
    normal, distance = get_plane_from_pose(rvec, tvec)

    matches = template_matching_pyramid(
        template,
        target,
        threshold=0.2,
        angle_range=(-45, 45),
        angle_step_coarse=3,  # Coarse search: every 5 degrees
        angle_step_fine=3,  # Fine search: every 3 degree
        max_matches=1,
        pyramid_levels=5,  # Use 5-level pyramid
    )
    for i, (x, y, w, h, score, angle) in enumerate(matches):
        # Calculate center of bounding box (u, v)
        u = x + w // 2
        v = y + h // 2
        # Convert pixel to 3D point on plane (in camera coordinates)
        P_cam = perform_pixel_to_plane(u, v, cameraMatrix, normal, distance)

        if P_cam is None:
            print("Warning: Could not convert pixel to 3D point")
            continue

        print(f"3D Point (camera frame): {P_cam.T}")

        # Get rotation matrix for the detected object
        # The angle is in-plane rotation on the table
        R_object = get_rotation_about_normal(normal, np.radians(angle))
        print(f"Rotation matrix of object:\n{R_object}")

        # Create full transformation matrix (camera to object)
        T_cam_template = convert_to_homogeneous(R_object, P_cam)
        return T_cam_template


def load_camera_calibration():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mtx_path = os.path.join(current_dir, "data", "calibration_matrix_usb.npy")
    dist_path = os.path.join(
        current_dir, "data", "distortion_coefficients_usb.npy"
    )
    cameraMatrix = np.load(mtx_path)
    distCoeffs = np.load(dist_path)
    return cameraMatrix, distCoeffs


def collect_points(T_base_template):
    collected_points = []
    id = 0
    while True:
        key = input("Press 'c' to collect point or 'q' to quit: ")
        if key.lower() == "c":
            error, pose = get_TCP_pose()
            T_base_target = convert_euler_to_homogeneous(
                pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
            )
            # T_template_target = (
            # np.linalg.inv(T_base_template) @ T_base_target)
            T_template_target = np.eye(4)
            delta = T_base_target[:3, 3] - T_base_template[:3, 3]
            T_template_target[:3, 3] = T_base_template[:3, 3].T @ delta
            collected_points.append(
                {"id": id, "T_template_target": T_template_target.tolist()}
            )
            print(f"Collected point ID {id}:")
            id += 1
        elif key.lower() == "q":
            print(f"\nCollected {len(collected_points)} points. Saving.")

            output_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data",
                "collected_points.json",
            )
            with open(output_path, "w") as f:
                json.dump(collected_points, f, indent=4)

            print(f"Points saved to {output_path}")
            break
        else:
            print("Invalid input. Please try again.")


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


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    T_base_gripper = calculate_T_base_gripper()
    hand_eye_path = os.path.join(current_dir, "data", "hand_eye_params.json")
    with open(hand_eye_path, "r") as f:
        hand_eye_params = json.load(f)
    T_gripper_cam = np.array(hand_eye_params["T_gripper_cam"])
    T_cam_template = calculate_T_cam_template(
        "base_03.jpg", "template_edge.png", "plane_2.jpg"
    )
    T_base_template = T_base_gripper @ T_gripper_cam @ T_cam_template
    pose = transform_to_pose6dof_deg(T_base_template)
    print(f"Template Pose [mm, deg]: {pose}")
    # Teaching
    collect_points(T_base_template)
