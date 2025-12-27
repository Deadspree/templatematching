import cv2
import numpy as np

# Global variables
drawing = False
deleting = False
roi_selected = False
ix, iy = -1, -1
roi = None
edges = None
img_display = None
edge_mask = None
original_img = None
brush_size = 10  # Default brush size


def on_brush_size_change(val):
    """Callback for brush size trackbar"""
    global brush_size
    brush_size = max(1, val)  # Ensure brush size is at least 1


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, roi_selected, roi, img_display, deleting
    global edge_mask, brush_size

    if not roi_selected:
        # ROI selection mode
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_temp = img_display.copy()
                cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Template Edge Extraction", img_temp)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            roi = (min(ix, x), min(iy, y), abs(x - ix), abs(y - iy))
            cv2.rectangle(
                img_display,
                (roi[0], roi[1]),
                (roi[0] + roi[2], roi[1] + roi[3]),
                (0, 255, 0),
                2,
            )
            cv2.imshow("Template Edge Extraction", img_display)
            roi_selected = True

    else:
        # Edge deletion mode (only if edges exist)
        if edges is not None:
            # Show brush cursor
            if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
                img_temp = img_display.copy()
                if deleting:
                    cv2.circle(edge_mask, (x, y), brush_size, 0, -1)
                    update_edge_display()
                    img_temp = img_display.copy()
                # Draw brush cursor
                cv2.circle(img_temp, (x, y), brush_size, (0, 0, 255), 2)
                cv2.imshow("Template Edge Extraction", img_temp)

            if event == cv2.EVENT_LBUTTONDOWN:
                deleting = True
                cv2.circle(edge_mask, (x, y), brush_size, 0, -1)
                update_edge_display()

            elif event == cv2.EVENT_LBUTTONUP:
                deleting = False


def update_edge_display():
    global img_display, edges, edge_mask
    img_display = cv2.cvtColor(edges * edge_mask, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Template Edge Extraction", img_display)


def reset_roi():
    global roi_selected, roi, edges, edge_mask, img_display, original_img
    roi_selected = False
    roi = None
    edges = None
    edge_mask = None
    img_display = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Template Edge Extraction", img_display)
    print("ROI reset. Draw a new ROI with mouse.")


def template_gui():
    global img_display, roi, edges, roi_selected, edge_mask, original_img, brush_size

    # Load image
    img_path = r"C:\Users\Admin_PC\Desktop\robot\Pattern_detection\input\template8.jpg"
    img = (
        cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if isinstance(img_path, str)
        else None
    )

    if img is None:
        print("Error: Could not load image")
        return

    img = cv2.medianBlur(img, 5)
    original_img = img.copy()
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Create window
    cv2.namedWindow("Template Edge Extraction")
    cv2.setMouseCallback("Template Edge Extraction", mouse_callback)

    # Create trackbar for brush size
    cv2.createTrackbar(
        "Brush Size", "Template Edge Extraction", 10, 100, on_brush_size_change
    )

    print("Instructions:")
    print("1. Draw ROI with mouse")
    print("2. Press 'r' to reset and reselect ROI")
    print("3. Press 'e' to detect edges")
    print("4. Use trackbar to adjust brush size")
    print("5. Use mouse to delete unwanted edges")
    print("6. Press 's' to save")
    print("7. Press 'q' to quit")

    while True:
        cv2.imshow("Template Edge Extraction", img_display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            # Reset ROI
            reset_roi()

        elif key == ord("e") and roi_selected and edges is None:
            # Extract ROI and detect edges
            x, y, w, h = roi
            roi_img = original_img[y : y + h, x : x + w]

            # Apply Canny edge detection
            edges = cv2.Canny(roi_img, 50, 150)
            edge_mask = np.ones_like(edges)

            img_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Template Edge Extraction", img_display)
            print("Edges detected. Use mouse to delete unwanted edges.")

        elif key == ord("s") and edges is not None:
            # Save edge image
            output_path = "template_edge.png"
            cv2.imwrite(output_path, edges * edge_mask)
            print(f"Edge template saved to {output_path}")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    template_gui()


if __name__ == "__main__":
    main()
