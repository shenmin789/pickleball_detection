import cv2
import numpy as np
import time
import threading
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

MODEL_PATH = 'runs/detect/train/weights/best.pt'
VIDEO_PATH = "pickleball_video_3.mp4"
CLASS_NAME = "ball"
# Add these variables at the top with other configurations
ROI_THICKNESS = 80
TOUCH_LOG_FILE = "ball_touches.csv"
touch_events = []  # To store touch events
CLIP_BUFFER_SECONDS = 2
CLIP_FPS = 30  # Should match your camera's actual FPS
# Add these new configurations
CALIBRATION_FRAMES = 15  # Number of frames to average ROI detection
ROI_MARGIN = 50         # Padding around detected line
STARTUP_DELAY = 2

model = YOLO(MODEL_PATH)

# Initialize DeepSort with your custom filter
tracker = DeepSort(max_age=15)

class TouchDetector:
    def __init__(self, confidence_frames=3):
        self.state_buffer = deque(maxlen=5)
        self.confidence_threshold = confidence_frames
        self.current_state = False
        
    def update(self, current_detection):
        self.state_buffer.append(current_detection)
        
        # Require 3/5 consecutive frames to confirm state change
        positive_count = sum(self.state_buffer)
        consensus = positive_count >= self.confidence_threshold
        
        if consensus != self.current_state:
            self.current_state = consensus
            return True  # State changed
        return False

class ClipManager:
    def __init__(self):
        self.buffer = deque(maxlen=CLIP_FPS * CLIP_BUFFER_SECONDS)
        self.recording = False
        self.post_event_frames = []
        self.current_event = None
        self.writing = False
        self.lock = threading.Lock()
        
    def add_frame(self, frame):
        with self.lock:
            self.buffer.append(frame.copy())
            
            if self.recording:
                self.post_event_frames.append(frame.copy())
                
                # Check if we've captured enough post-event frames
                if len(self.post_event_frames) >= CLIP_FPS * CLIP_BUFFER_SECONDS:
                    self.save_clip()
    
    def start_recording(self, event_time):
        with self.lock:
            if not self.recording:
                self.recording = True
                self.current_event = {
                    'pre_frames': list(self.buffer),
                    'post_frames': [],
                    'start_time': event_time
                }
    
    def save_clip(self):
        def write_clip(pre, post, event_time):
            if len(pre) == 0 and len(post) == 0:
                return
                
            filename = f"clip_{event_time.strftime('%Y%m%d_%H%M%S_%f')}.mp4"
            height, width = pre[0].shape[:2] if pre else post[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, CLIP_FPS, (width, height))
            
            for frame in pre + post:
                out.write(frame)
            
            out.release()
        
        with self.lock:
            self.writing = True
            event_time = self.current_event['start_time']
            all_frames = self.current_event['pre_frames'] + self.post_event_frames
            
            # Start writing in a new thread
            threading.Thread(target=write_clip, 
                           args=(self.current_event['pre_frames'],
                                 self.post_event_frames,
                                 event_time)).start()
            
            # Reset recording state
            self.recording = False
            self.post_event_frames = []
            self.current_event = None
            self.writing = False

class CourtCalibrator:
    def __init__(self):
        self.roi_rect = None
        self.line_points = deque(maxlen=CALIBRATION_FRAMES)
        self.merged_line = None
        self.calibration_done = False
        self.calibration_started = False

    def start_calibration(self, frame):
        if not self.calibration_started:
            self.calibration_started = True
            print(f"Starting calibration in {STARTUP_DELAY} seconds...")
            time.sleep(STARTUP_DELAY)
            
        merged_line = detect_and_merge_horizontal_lines(frame)
        if merged_line is not None:
            x1, y, x2, _ = merged_line
            self.line_points.append((x1, y, x2))
            
            if len(self.line_points) == CALIBRATION_FRAMES:
                self._finalize_roi()
                self.calibration_done = True
                return True
        return False
        
    def calibrate(self, frame):
        """Detect court line once and store ROI coordinates"""
        merged_line = detect_and_merge_horizontal_lines(frame)
        if merged_line is not None:
            self.merged_line = merged_line
            x1, y, x2, _ = merged_line
            self.line_points.append((x1, y, x2))
            
            if len(self.line_points) == CALIBRATION_FRAMES:
                self._finalize_roi()
                return True
        return False
    
    def _finalize_roi(self):
        """Average line positions from multiple frames"""
        avg_x1 = np.mean([p[0] for p in self.line_points])
        avg_y = np.mean([p[1] for p in self.line_points])
        avg_x2 = np.mean([p[2] for p in self.line_points])
        
        self.roi_rect = (
            int(avg_x1 - ROI_MARGIN),
            int(avg_y - ROI_MARGIN),
            int(avg_x2 + ROI_MARGIN),
            int(avg_y + ROI_MARGIN)
        )

class HorizontalLineDetector:
    def __init__(self, horizontal_tolerance=10, min_line_length=50, max_line_gap=10, threshold_value=200, smoothing_factor=0.8):
        """
        Initialize the detector with parameters.
        smoothing_factor (0-1): higher values give more weight to previous estimates.
        """
        self.horizontal_tolerance = horizontal_tolerance
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.threshold_value = threshold_value
        self.smoothing_factor = smoothing_factor
        self.prev_line = None  # Store previous detected line for temporal smoothing

    def detect_and_merge_horizontal_lines(self, frame):
        """
        Detects horizontal white line segments in an image and merges segments that are nearly collinear.
        Uses grouping and temporal smoothing to produce a stable output.
        Returns a merged horizontal line as (min_x, y, max_x, y) or None if no line is detected.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optionally, apply a Gaussian blur to reduce noise
        # gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Threshold to emphasize white areas
        _, thresh = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        
        # Apply morphological closing to connect broken segments
        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Apply Canny edge detection
        edges = cv2.Canny(closed, 50, 150, apertureSize=3)
        
        # Detect line segments using Probabilistic Hough Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                                minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        if lines is None:
            return None

        # Filter for nearly horizontal segments (vertical difference is small)
        horizontal_segments = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < self.horizontal_tolerance:
                horizontal_segments.append((x1, y1, x2, y2))
        
        if not horizontal_segments:
            return None

        # Group segments by similar average y coordinate
        groups = []
        for seg in horizontal_segments:
            x1, y1, x2, y2 = seg
            avg_y = (y1 + y2) / 2.0
            added = False
            # Group segments that lie on a similar horizontal level
            for group in groups:
                if abs(group['avg_y'] - avg_y) < self.horizontal_tolerance:
                    group['segments'].append(seg)
                    # Recalculate group's average y
                    group['avg_y'] = np.mean([ (s[1] + s[3]) / 2.0 for s in group['segments'] ])
                    added = True
                    break
            if not added:
                groups.append({'avg_y': avg_y, 'segments': [seg]})
        
        # Choose the group with the highest average y (closest to the camera)
        best_group = max(groups, key=lambda g: g['avg_y'])
        
        # Merge segments in the best group:
        # The merged line spans from the smallest x to the largest x among all segments,
        # and uses the group's average y as the stable y coordinate.
        min_x = min(min(seg[0], seg[2]) for seg in best_group['segments'])
        max_x = max(max(seg[0], seg[2]) for seg in best_group['segments'])
        merged_y = int(round(best_group['avg_y']))
        current_line = (min_x, merged_y, max_x, merged_y)

        # Apply temporal smoothing if a previous line exists.
        if self.prev_line is not None:
            smoothed_line = tuple(int(self.smoothing_factor * p_prev + (1 - self.smoothing_factor) * p_curr)
                                  for p_prev, p_curr in zip(self.prev_line, current_line))
        else:
            smoothed_line = current_line

        # Update the previous line and return the smoothed result.
        self.prev_line = smoothed_line
        return smoothed_line

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

def detect_and_merge_horizontal_lines_2(frame, horizontal_tolerance=10, min_line_length=50, max_line_gap=10, threshold_value=200):
    """
    Detects horizontal white line segments in an image and merges segments that are nearly collinear.
    This approach improves stability by grouping segments by their average y coordinate and merging them.
    
    The processing steps include:
      - Converting the frame to grayscale.
      - Thresholding to emphasize white areas.
      - (Optional) Morphological closing to connect broken segments.
      - Canny edge detection.
      - Detecting line segments using the Probabilistic Hough Transform.
      - Filtering out segments that are not nearly horizontal.
      - Grouping segments based on similar average y values.
      - Selecting the group with the highest average y (nearest to the camera).
      - Merging segments by taking the minimum and maximum x coordinates and the averaged y value.
    
    Parameters:
        frame (numpy.ndarray): Input image in BGR format.
        horizontal_tolerance (int): Maximum y-difference to consider segments nearly horizontal.
        min_line_length (int): Minimum length of a line segment to be considered.
        max_line_gap (int): Maximum allowed gap between segments in HoughLinesP.
        threshold_value (int): Threshold for binarizing the image (default 200 for bright white lines).
    
    Returns:
        tuple or None: A merged horizontal line as (min_x, y, max_x, y), or None if no line is detected.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Optionally, apply a Gaussian blur to reduce noise (uncomment if needed)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to emphasize white areas
    _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Optional: Apply morphological closing to connect broken segments
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Apply Canny edge detection
    edges = cv2.Canny(closed, 50, 150, apertureSize=3)
    
    # Detect line segments using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is None:
        return None

    # Filter for nearly horizontal segments (vertical difference is small)
    horizontal_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < horizontal_tolerance:
            horizontal_segments.append((x1, y1, x2, y2))
    
    if not horizontal_segments:
        return None

    # Group segments by similar average y coordinate
    groups = []
    for seg in horizontal_segments:
        x1, y1, x2, y2 = seg
        avg_y = (y1 + y2) / 2.0
        added = False
        # Try to add to an existing group if the average y is close enough
        for group in groups:
            if abs(group['avg_y'] - avg_y) < horizontal_tolerance:
                group['segments'].append(seg)
                # Recalculate the group's average y using all segments in the group
                group['avg_y'] = np.mean([ (s[1] + s[3]) / 2.0 for s in group['segments'] ])
                added = True
                break
        if not added:
            groups.append({'avg_y': avg_y, 'segments': [seg]})
    
    # Choose the group with the highest average y (closest to the camera)
    best_group = max(groups, key=lambda g: g['avg_y'])
    
    # Merge segments in the best group:
    # The merged line spans from the smallest x to the largest x among all segments,
    # and uses the group's average y as the stable y coordinate.
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

def is_ball_touching_roi(ball_box, roi_rect):
    """
    Check if the ball's bounding box intersects with the ROI rectangle
    ball_box: (x1, y1, x2, y2) in pixel coordinates
    roi_rect: (left, top, right, bottom) coordinates of the ROI
    """
    # Unpack coordinates
    bx1, by1, bx2, by2 = ball_box
    rx1, ry1, rx2, ry2 = roi_rect
    
    # Calculate intersection area
    x_overlap = max(0, min(bx2, rx2) - max(bx1, rx1))
    y_overlap = max(0, min(by2, ry2) - max(by1, ry1))
    
    return x_overlap > 0 and y_overlap > 0

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    class_names = model.names
    touch_detector = TouchDetector()
    calibrator = CourtCalibrator()
    clip_manager = ClipManager()
    calibration_done = False
    touch_state = False
    last_event_time = None
    main_window_created = False
    detector = HorizontalLineDetector(smoothing_factor=0.8)

    # Initialize touch log file
    with open(TOUCH_LOG_FILE, 'w') as f:
        f.write("timestamp,track_id,event_type,consensus\n")

    # Initialize clip buffer thread
    # def buffer_frames():
    #     while cap.isOpened():
    #         ret, frame = cap.read()
    #         if ret:
    #             frame = cv2.rotate(frame, cv2.ROTATE_180)
    #             clip_manager.add_frame(frame)

    # buffer_thread = threading.Thread(target=buffer_frames)
    # buffer_thread.start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # rotate frame 
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # # Detect and merge horizontal white line segments
        merged_line = detector.detect_and_merge_horizontal_lines(frame)
        roi_rect = None

        if merged_line is not None:
            x1, y, x2, _ = merged_line
            margin = ROI_THICKNESS // 2
            roi_rect = (
                x1,                  # left
                y - margin,          # top
                x2,                  # right
                y + margin           # bottom
            )
            draw_roi_around_line(frame, merged_line, ROI_THICKNESS)
        else:
            cv2.putText(frame, "No horizontal white line detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # YOLO detection with motion-friendly settings
        results = model.predict(frame, conf=0.3, imgsz=640, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        # Filter for "Balls" class
        ball_indices = [i for i, c in enumerate(class_ids) if class_names[c] == CLASS_NAME]
        ball_boxes = boxes[ball_indices]
        ball_confidences = confidences[ball_indices]

        # Update DeepSort tracker
        detections = [(box, conf, None) for box, conf in zip(ball_boxes, ball_confidences)]
        tracks = tracker.update_tracks(detections, frame=frame)

        # Match tracks to current detections and draw YOLO boxes with track IDs
        for det_idx, (box, conf, _) in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            ball_box = (x1, y1, x2, y2)
            
            # Find the track ID for this detection (if matched)
            for track in tracks:
                track_box = track.to_ltrb()
                tx1, ty1, tx2, ty2 = map(int, track_box)
                # Check if track box overlaps significantly with detection box
                overlap = (x1 < tx2) and (x2 > tx1) and (y1 < ty2) and (y2 > ty1)
                
                if overlap and track.is_confirmed() and roi_rect is not None:
                    track_id = track.track_id
                    touching = is_ball_touching_roi(ball_box, roi_rect)

                    state_changed = touch_detector.update(touching)
                    
                    # Log touch events
                    if state_changed:
                        # clip_manager.start_recording(current_time)
                        touch_events.append({
                            'timestamp': current_time,
                            'track_id': track_id,
                            'event_type': 'TOUCH_START' if touch_detector.current_state else 'TOUCH_END',
                            'consensus': touch_detector.current_state
                        })

                    
                    # Change box color if touching
                    color = (0, 0, 255) if touching else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Save touch events to file periodically
        if len(touch_events) > 0:
            print(touch_events)
            with open(TOUCH_LOG_FILE,  mode="a", newline="") as f:
                for event in touch_events:
                    f.write(f"{event['timestamp']},{event['track_id']},{event['event_type']},{event['consensus']}\n")
            touch_events.clear()
        
        cv2.imshow("Pickleball Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        

    cap.release()
    cv2.destroyAllWindows()
    # buffer_thread.join()

if __name__ == "__main__":
    main()