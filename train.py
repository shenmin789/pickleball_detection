from ultralytics import YOLO
if __name__ == "__main__":
    # Load YOLO model
    # Load YOLOv8 model (pretrained on COCO)
    model = YOLO("yolov8n.pt")  # 'n' = nano model (fastest), change to 's', 'm', 'l', or 'x' for more accuracy
    # model = YOLO("runs/detect/train/weights/best.pt")

    # Train the model on your dataset
    # model.train(data="pickleball_dataset_2/data.yaml", epochs=150, imgsz=640, batch=16, device=0)

    #fine tuning the model
    # model.train(data="dataset_1/data.yaml",epochs=150,imgsz=640,batch=16)
    model.train(
        data="dataset/data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        device=0
    )
