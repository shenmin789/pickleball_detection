import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize models
ball_model = YOLO("yolov8n.pt")  # Pre-trained on COCO (class 32 = sports ball)
tracker = DeepSort(max_age=15)    # Ball tracker
