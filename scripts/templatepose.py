# External libary
import cv2
import numpy as np

# Internal library
from scripts.invariant_edge_matching import template_matching_pyramid


def charuco_pose_estimation(
    image: np.ndarray,
    matrix_coefficients_path: str,
    distortion_coefficients_path: str,
    marker_length: float = 0.022,
    square_length: float = 0.03,
    aruco_dict_type: int = cv2.aruco.DICT_6X6_100,
):
    """
    !Charuco board pose estimation
    @image (np.ndarray): input image
    @matrix_coefficients_path (str): path to matrix coefficient file
    @distortion_coefficients_path (str): path to distortion coefficient file
    @marker_length (float): length of marker in metres
    @square_length (float): length of each square in metres
    @aruco_dict_type: Type of dictionary for available aruco dictionary
    """
    cameraMatrix = np.load(matrix_coefficients_path)
    distCoeffs = np.load(distortion_coefficients_path)
    squareLength = square_length
    markerLength = marker_length
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

    board = cv2.aruco.CharucoBoard(
        (7, 5), squareLength, markerLength, dictionary
    )

    # ---- ArUco marker detection ----
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    marker_corners, marker_ids, rejected = detector.detectMarkers(image)
    # ---- Charuco detection ----
    charuco_detector = cv2.aruco.CharucoDetector(board)
    (
        charuco_corners,
        charuco_ids,
        charuco_marker_corners,
        charuco_marker_ids,
    ) = charuco_detector.detectBoard(
        image, markerCorners=marker_corners, markerIds=marker_ids
    )
    rvec = None
    tvec = None
    if charuco_ids is not None and len(charuco_corners) > 0:

        cv2.aruco.drawDetectedCornersCharuco(
            image, charuco_corners, charuco_ids
        )

        # ---- SolvePnP using Charuco corners ----
        objPoints, imgPoints = board.matchImagePoints(
            charuco_corners, charuco_ids
        )

        if len(objPoints) >= 15:
            valid, rvec, tvec = cv2.solvePnP(
                objPoints, imgPoints, cameraMatrix, distCoeffs
            )
            cv2.solvePnPRefineLM(
                objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec
            )
            cv2.drawFrameAxes(
                image, cameraMatrix, distCoeffs, rvec, tvec, squareLength * 3
            )
    return image, rvec, tvec


def new_charuco_pose_estimation(
    image, cameraMatrix, distCoeffs, board, visualize=False
):
    """
    !Charuco board pose estimation
    @image (np.ndarray): input image
    @cameraMatrix (np.ndarray): camera matrix from calibration
    @distCoeffs (np.ndarray): distortion coefficients from calibration
    @board (cv2.aruco.CharucoBoard): Charuco board object
    """
    # ---- ArUco marker detection ----
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dictionary = board.getDictionary()
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    marker_corners, marker_ids, rejected = detector.detectMarkers(image)

    # ---- Charuco detection ----
    charuco_detector = cv2.aruco.CharucoDetector(board)
    (
        charuco_corners,
        charuco_ids,
        charuco_marker_corners,
        charuco_marker_ids,
    ) = charuco_detector.detectBoard(
        image, markerCorners=marker_corners, markerIds=marker_ids
    )
    rvec = None
    tvec = None
    if charuco_ids is not None and len(charuco_corners) > 0:

        cv2.aruco.drawDetectedCornersCharuco(
            image, charuco_corners, charuco_ids
        )

        # ---- SolvePnP using Charuco corners ----
        objPoints, imgPoints = board.matchImagePoints(
            charuco_corners, charuco_ids
        )

        if len(objPoints) >= 15:
            valid, rvec, tvec = cv2.solvePnP(
                objPoints, imgPoints, cameraMatrix, distCoeffs
            )
            cv2.solvePnPRefineLM(
                objPoints, imgPoints, cameraMatrix, distCoeffs, rvec, tvec
            )
            proj, _ = cv2.projectPoints(
                objPoints, rvec, tvec, cameraMatrix, distCoeffs
            )
            reproj_err = np.mean(
                np.linalg.norm(proj.squeeze() - imgPoints.squeeze(), axis=1)
            )
            print("High reprojection error:", reproj_err)
            if reproj_err > 0.7:
                return image, None, None
            if visualize:
                cv2.drawFrameAxes(
                    image, cameraMatrix, distCoeffs, rvec, tvec, 0.03 * 3
                )
                cv2.aruco.drawDetectedCornersCharuco(
                    image, charuco_corners, charuco_ids
                )

                cv2.putText(
                    image,
                    f"Reproj Error: {reproj_err:.3f}px",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    image,
                    "Adjust the camera so that error is less than 0.3",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
    return image, rvec, tvec


def get_plane_from_pose(rvec, tvec):
    """
    ! Get normal vector to table plane and distance from camera
    @param rvec<np.ndarray>: Rotation vector from pose estimation
    @param tvec<np.ndarray>: Translation vector from pose estimation
    """
    R, _ = cv2.Rodrigues(rvec)

    # Board Z axis in camera frame
    normal = R[:, 2].reshape(3, 1)
    normal /= np.linalg.norm(normal)

    # Ensure normal faces the camera (+Z_cam)
    camera_z = np.array([[0.0], [0.0], [1.0]])
    if normal.T @ camera_z < 0:
        normal = -normal
        tvec = -tvec  # optional but recommended for consistency

    distance = -float(normal.T @ tvec)
    return normal, distance


def perform_pixel_to_plane(u, v, K, n, d):
    """
    ! Convert pixel coordinates to 3D coordinates on a plane
    @param u<int>: Pixel x-coordinates
    @param v<int>: Pixel y-coordinates
    @param K<np.ndarray>: Camera intrinsic matrix
    @param n<np.ndarray>: Normal vector of the plane
    @param d<float>: Distance from camera to plane along normal vector
    """
    pix = np.array([u, v, 1]).reshape(3, 1)
    ray = np.linalg.inv(K) @ pix
    ray = ray / np.linalg.norm(ray)

    denominator = float(n.T @ ray)

    t = -d / denominator
    P_cam = t * ray
    error_plane = n.T @ P_cam + d
    print("Plane residual:", error_plane)
    return P_cam


def get_rotation_about_normal(n, theta):
    """
    ! Get rotation matrix for rotation around a given axis
    @param n<np.ndarray>: Axis of rotation (3x1 vector)
    @param theta<float>: Rotation angle in radians"""
    axis = n.flatten()
    axis = axis / np.linalg.norm(axis)
    # Enforce consistent axis direction (toward camera)
    if axis[2] < 0:
        axis = -axis

    # Template-matching angles are usually image-CW → invert
    theta = -theta

    rvec = axis * theta
    R, _ = cv2.Rodrigues(rvec)

    return R


def convert_to_homogeneous(R, t):
    """
    ! Convert rotation matrix and translation vector to homogeneous
    transformation matrix
    @param R<np.ndarray>: Rotation matrix (3x3)
    @param t<np.ndarray>: Translation vector (3x1)
    """
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.flatten()
    return T


def main():
    """
    Main function to perform template matching and convert results to 3D 
    coordinates
    """
    # Load calibration data
    matrix_path = (
        r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\calibration\calibration_matrix_usb.npy"
    )
    distortion_path = (
        r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\calibration\distortion_coefficients_usb.npy"
    )

    cameraMatrix = np.load(matrix_path)
    distCoeffs = np.load(distortion_path)

    # Create Charuco board
    squareLength = 0.03
    markerLength = 0.022
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
    board = cv2.aruco.CharucoBoard(
        (7, 5), squareLength, markerLength, dictionary
    )

    # Load target image
    target_path = (
        r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\input\2d_01.jpg"
    )
    plane_path = r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\input\plane_calib_4.jpg"
    plane = cv2.imread(plane_path)
    target = cv2.imread(target_path)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    print("=== Step 1: Charuco Pose Estimation ===")
    # Perform pose estimation to get plane parameters
    pose_image, rvec, tvec = new_charuco_pose_estimation(
        plane.copy(), cameraMatrix, distCoeffs, board
    )
    print("rvec:", rvec)
    print("tvec:", tvec)
    if rvec is None or tvec is None:
        print("Error: Could not estimate Charuco board pose")
        return

    # Get plane normal and distance
    normal, distance = get_plane_from_pose(rvec, tvec)

    # Get rotation matrix for board coordinate frame
    R_board, _ = cv2.Rodrigues(rvec)

    print("\n=== Step 2: Template Matching ===")
    # Load template
    template_path = r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\scripts\template_edge.png"
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if template is None:
        print("Error: Could not load template")
        return

    # Extract edges from target
    target_edges = cv2.Canny(gray, 20, 300)

    # Perform template matching
    matches = template_matching_pyramid(
        template,
        target_edges,
        threshold=0.2,
        angle_range=(-45, 45),
        angle_step_coarse=3,
        angle_step_fine=3,
        max_matches=1,
        pyramid_levels=5,
    )

    print("\n=== Step 3: 2D to 3D Transformation ===")
    results_3d = []

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
        T_cam_to_obj = convert_to_homogeneous(R_object, P_cam)

        # Store results
        result = {
            "match_id": i + 1,
            "2d_center": (int(u), int(v)),
            "2d_bbox": (int(x), int(y), int(w), int(h)),
            "score": float(score),
            "angle_deg": float(angle),
            "3d_position_camera": P_cam.flatten().tolist(),
            "transformation_matrix": T_cam_to_obj.tolist(),
        }
        results_3d.append(result)

    print("\n=== Step 4: Save Results ===")
    # Save to JSON
    import json

    json_path = r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\output\data\3d_results.json"
    with open(json_path, "w") as f:
        json.dump(results_3d, f, indent=4)
    print(f"3D results saved to: {json_path}")


if __name__ == "__main__":
    main()
