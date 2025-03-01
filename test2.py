import cv2
import numpy as np
import time
from datetime import datetime
from collections import deque
from ultralytics import YOLO

# Test Configuration
TEST_VIDEO = {'id': 'corner_view', 'path': 'pickleball_video_3.mp4'}

SKIP_FRAMES = 0
STATE_BUFFER_SIZE = 3
LOG_FILE = 'test_results.csv'
MODEL_PATH = 'runs/detect/train/weights/best.pt'
SHOW_DEBUG = True  # Enable visual debugging
MANUAL_FALLBACK = True

# Load ML model
model = YOLO(MODEL_PATH)

def detect_and_merge_horizontal_lines(frame, horizontal_tolerance=10, min_line_length=50):
    """
    Detects horizontal white line segments and merges segments that are nearly collinear.
    This helps combine segments that may be split by the middle vertical line.
    Returns a merged horizontal line as (min_x, y, max_x, y) or None if no line is detected.
    """
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold to emphasize white areas (assuming the lines are bright)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Apply Canny edge detection on the thresholded image
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    
    # Detect line segments using the Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=50, 
                            minLineLength=min_line_length, 
                            maxLineGap=10)
    if lines is None:
        return None

    # Filter for nearly horizontal segments (difference in y is small)
    horizontal_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < horizontal_tolerance:
            horizontal_segments.append((x1, y1, x2, y2))
    
    if not horizontal_segments:
        return None

    # Group segments by their average y coordinate
    groups = []
    for seg in horizontal_segments:
        x1, y1, x2, y2 = seg
        avg_y = (y1 + y2) / 2.0
        added = False
        # Group segments that lie on a similar horizontal level
        for group in groups:
            if abs(group['avg_y'] - avg_y) < horizontal_tolerance:
                group['segments'].append(seg)
                # Recalculate group average y (simple average)
                group['avg_y'] = (group['avg_y'] * len(group['segments']) + avg_y) / (len(group['segments']) + 1)
                added = True
                break
        if not added:
            groups.append({'avg_y': avg_y, 'segments': [seg]})
    
    # Choose the group with the highest average y value (nearest to the camera)
    best_group = max(groups, key=lambda g: g['avg_y'])
    
    # Merge the segments in the chosen group:
    #   - The merged line will span from the smallest x to the largest x of all segments
    #   - Use the average y coordinate of the group as the line's y-value
    min_x = min(min(seg[0], seg[2]) for seg in best_group['segments'])
    max_x = max(max(seg[0], seg[2]) for seg in best_group['segments'])
    merged_y = int(round(best_group['avg_y']))
    
    return (min_x, merged_y, max_x, merged_y)

def draw_roi_around_line(frame, line, roi_thickness=20):
    """
    Draws the merged horizontal line and a rectangular ROI around it.
    The ROI is a horizontal band with a total thickness of roi_thickness pixels.
    """
    x1, y, x2, _ = line
    margin = roi_thickness // 2
    left = x1
    right = x2
    top = y - margin
    bottom = y + margin
    
    # Draw the ROI rectangle (green)
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    # Draw the detected horizontal line (red)
    cv2.line(frame, (x1, y), (x2, y), (0, 0, 255), 2)
    return frame

def main():
    cap = cv2.VideoCapture(TEST_VIDEO.get("path"))  # Use your camera index or video file
    if not cap.isOpened():
        print("Error: Unable to open video stream")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        # Detect and merge horizontal white line segments
        merged_line = detect_and_merge_horizontal_lines(frame)
        if merged_line is not None:
            frame = draw_roi_around_line(frame, merged_line, roi_thickness=100)
        else:
            cv2.putText(frame, "No horizontal white line detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Frame with ROI", frame)
        key = cv2.waitKey(1)
        if key == 27:  # Exit on pressing 'Esc'
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

