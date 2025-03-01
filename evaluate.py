from ultralytics import YOLO

if __name__ == "__main__":
    # Load trained model
    model = YOLO("runs/detect/train/weights/best.pt")

    
    # metrics = model.val(split="val")  
    # print(f"Ball mAP@0.5: {metrics.box.map50} | Precision: {metrics.box.p} | Recall: {metrics.box.r}")
    model.predict("test_image/test_img_1.jpg", conf=0.5, save=True)
