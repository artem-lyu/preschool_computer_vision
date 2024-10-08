from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolo11l-cls.pt')
    results = model.train(data="data", epochs=25, imgsz=900, patience=3)