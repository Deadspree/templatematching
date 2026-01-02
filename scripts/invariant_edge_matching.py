import cv2
import numpy as np
import json
import time


def load_images():
    """Load template and target images"""
    # Load template edge image
    template_path = r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\input\template_edge.png"
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    # Load target image
    target_path = (
        r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\input\target_22.jpg"
    )
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.medianBlur(target, 5)

    if template is None:
        print("Error: Could not load template image")
        return None, None

    if target is None:
        print("Error: Could not load target image")
        return None, None

    return template, target


def extract_edges(image, low_threshold=20, high_threshold=300):
    """Extract edges from target image"""
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


def rotate_image(image, angle):
    """
    Rotate image by given angle

    Args:
        image: Input image
        angle: Rotation angle in degrees

    Returns:
        Rotated image
    """
    h, w = image.shape
    center = (w // 2, h // 2)

    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust rotation matrix for new center
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform rotation
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated


def create_pyramid(image, levels=3, scale_factor=2):
    """
    Create image pyramid

    Args:
        image: Input image
        levels: Number of pyramid levels
        scale_factor: Downsampling factor between levels

    Returns:
        List of images from coarse to fine
    """
    pyramid = [image]
    current = image.copy()

    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        pyramid.append(current)

    return pyramid[::-1]  # Return from coarse to fine


def template_matching_rotation_invariant(
    template,
    target_edges,
    method=cv2.TM_CCOEFF_NORMED,
    threshold=0.4,
    angle_range=(-180, 180),
    angle_step=1,
    max_matches=None,
):
    """
    Perform rotation-invariant template matching

    Args:
        template: Template edge image
        target_edges: Target edge image
        method: OpenCV matching method
        threshold: Matching threshold (0-1)
        angle_range: Tuple of (min_angle, max_angle) in degrees
        angle_step: Step size for angle rotation
        max_matches: Maximum number of matches to return (None for unlimited)

    Returns:
        List of matching locations (x, y, width, height, score, angle)
    """
    all_matches = []
    angles = np.arange(angle_range[0], angle_range[1], angle_step)

    print(f"Searching through {len(angles)} rotation angles...")

    for angle in angles:
        # Rotate template
        rotated_template = rotate_image(template, angle)
        h, w = rotated_template.shape

        # Skip if rotated template is larger than target
        if h > target_edges.shape[0] or w > target_edges.shape[1]:
            continue

        # Perform template matching
        result = cv2.matchTemplate(target_edges, rotated_template, method)

        # Find locations above threshold
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):
            score = result[pt[1], pt[0]]
            all_matches.append((pt[0], pt[1], w, h, score, angle))

        # Print progress every 10 degrees
        if angle % 10 == 0:
            print(
                f"  Angle: {angle}° - Found {len(all_matches)} candidates"
                " so far"
            )

    # Apply non-maximum suppression across all angles
    all_matches = non_max_suppression_with_rotation(
        all_matches, overlap_thresh=0.3
    )

    # Limit to max_matches if specified
    if max_matches is not None and len(all_matches) > max_matches:
        # Sort by score (descending) and keep top N
        all_matches = sorted(all_matches, key=lambda x: x[4], reverse=True)[
            :max_matches
        ]
        print(f"Limited to top {max_matches} matches")

    return all_matches


def template_matching_pyramid(
    template,
    target_edges,
    method=cv2.TM_CCOEFF_NORMED,
    threshold=0.3,
    angle_range=(-90, 90),
    angle_step_coarse=5,  # Faster search at coarse level
    angle_step_fine=1,  # Precise search at fine level
    max_matches=1,
    pyramid_levels=3,
):
    """
    Pyramid-based rotation-invariant template matching

    Args:
        template: Template edge image
        target_edges: Target edge image
        method: OpenCV matching method
        threshold: Matching threshold (0-1)
        angle_range: Tuple of (min_angle, max_angle) in degrees
        angle_step_coarse: Step size for coarse level angle rotation
        angle_step_fine: Step size for fine level angle rotation
        max_matches: Maximum number of matches to return
        pyramid_levels: Number of pyramid levels

    Returns:
        List of matching locations (x, y, width, height, score, angle)
    """
    # Create pyramids
    template_pyramid = create_pyramid(template, pyramid_levels)
    target_pyramid = create_pyramid(target_edges, pyramid_levels)

    print(f"Created {pyramid_levels}-level pyramids")

    # Level 0: Coarse search with large angle steps
    print(f"\n=== Level 0 (Coarse): Searching every {angle_step_coarse}°")
    coarse_matches = template_matching_rotation_invariant(
        template_pyramid[0],
        target_pyramid[0],
        method=method,
        threshold=threshold,  # Lower threshold for coarse
        angle_range=angle_range,
        angle_step=angle_step_coarse,
        max_matches=(
            max_matches * 3 if max_matches else None
        ),  # Keep more candidates
    )

    if not coarse_matches:
        print("No matches found at coarse level")
        return []

    # Extract promising angles from coarse matches
    candidate_angles = [angle for _, _, _, _, _, angle in coarse_matches]

    # Level 1+: Refine around candidate angles
    refined_matches = []
    scale = 2 ** (pyramid_levels - 1)

    for level in range(1, pyramid_levels):
        scale = scale // 2
        print(
            f"\n=== Level {level}: Refining {len(candidate_angles)} candidates"
        )

        level_matches = []
        for base_angle in candidate_angles:
            # Search in narrow range around candidate angle
            angle_window = angle_step_coarse * (
                2 ** (pyramid_levels - level - 1)
            )
            local_range = (
                max(angle_range[0], base_angle - angle_window),
                min(angle_range[1], base_angle + angle_window),
            )

            matches = template_matching_rotation_invariant(
                template_pyramid[level],
                target_pyramid[level],
                method=method,
                threshold=(
                    threshold
                    if level == pyramid_levels - 1
                    else threshold - 0.05
                ),
                angle_range=local_range,
                angle_step=(
                    angle_step_fine
                    if level == pyramid_levels - 1
                    else angle_step_coarse // 2
                ),
                max_matches=max_matches * 2 if max_matches else None,
            )

            # Scale coordinates back to original size
            for x, y, w, h, score, angle in matches:
                level_matches.append(
                    (x * scale, y * scale, w * scale, h * scale, score, angle)
                )

        # NMS and update candidates
        level_matches = non_max_suppression_with_rotation(level_matches)
        candidate_angles = [angle for _, _, _, _, _, angle in level_matches]
        refined_matches = level_matches

        if not refined_matches:
            break

    # Final NMS and limit
    refined_matches = non_max_suppression_with_rotation(refined_matches)
    if max_matches and len(refined_matches) > max_matches:
        refined_matches = sorted(
            refined_matches, key=lambda x: x[4], reverse=True
        )[:max_matches]

    return refined_matches


def non_max_suppression_with_rotation(boxes, overlap_thresh=0.3):
    """
    Apply non-maximum suppression with rotation awareness

    Args:
        boxes: List of (x, y, w, h, score, angle)
        overlap_thresh: Overlap threshold for suppression

    Returns:
        Filtered list of boxes
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(scores)[::-1]

    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        overlap = (w * h) / area[idxs[1:]]

        idxs = np.delete(
            idxs,
            np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)),
        )

    result = []
    for box in boxes[keep]:
        result.append(
            (
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                float(box[4]),
                float(box[5]),
            )
        )

    return result


def draw_matches_with_rotation(image, matches, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes for matches with rotation angle"""
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i, (x, y, w, h, score, angle) in enumerate(matches):
        # Draw axis-aligned green rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)

        # Draw center point
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(result, (center_x, center_y), 5, (0, 0, 255), -1)

        # Draw rotated red rectangle
        # Calculate the four corners of the rotated rectangle
        rect = (
            (center_x, center_y),
            (w, h),
            -angle,
        )  # Negative angle for OpenCV convention
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(result, [box], 0, (0, 0, 255), thickness)

        # Add match info text
        text = f"#{i+1}: {score:.3f}, {angle} o"
        cv2.putText(
            result,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return result


def main():
    # Load images
    template, target = load_images()
    if template is None or target is None:
        return

    print(f"Template size: {template.shape}")
    print(f"Target size: {target.shape}")

    # Extract edges from target
    print("Extracting edges from target image...")

    target_edges = extract_edges(target)
    start = time.time()
    # Perform pyramid-based rotation-invariant template matching
    print("\nPerforming pyramid-based rotation-invariant template matching...")
    matches = template_matching_pyramid(
        template,
        target_edges,
        threshold=0.2,
        angle_range=(-45, 45),
        angle_step_coarse=3,  # Coarse search: every 5 degrees
        angle_step_fine=3,  # Fine search: every 3 degree
        max_matches=1,
        pyramid_levels=5,  # Use 5-level pyramid
    )
    elapsed_time = time.time() - start
    print(f"\nMatching completed in {elapsed_time:.2f} seconds")
    print(f"\nFound {len(matches)} matches:")
    for i, (x, y, w, h, score, angle) in enumerate(matches):
        print(
            f"Match {i+1}: Position=({x}, {y}), Size=({w}x{h}),"
            f" Score={score:.3f}, Angle={angle}°"
        )

    # Save matches to JSON file
    matches_data = [
        {"x": int(x), "y": int(y), "angle": float(angle)}
        for i, (x, y, w, h, score, angle) in enumerate(matches)
    ]

    json_output_path = r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\output\data\target_22.json"
    with open(json_output_path, "w") as json_file:
        json.dump(matches_data, json_file, indent=4)
    print(f"\nMatches saved to JSON: {json_output_path}")

    # Visualize results
    result_img = draw_matches_with_rotation(target, matches)

    # Show template and target edges for comparison
    template_display = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    target_edges_display = cv2.cvtColor(target_edges, cv2.COLOR_GRAY2BGR)

    # Display results
    cv2.namedWindow("Template", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Target Edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Matching Results", cv2.WINDOW_NORMAL)

    cv2.imshow("Template", template_display)
    cv2.imshow("Target Edges", target_edges_display)
    cv2.imshow("Matching Results", result_img)

    # Save results
    output_path = r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\output\images\target_22.png"
    cv2.imwrite(output_path, result_img)
    print(f"\nResults saved to {output_path}")

    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
