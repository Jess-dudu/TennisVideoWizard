from ultralytics import YOLO
from pathlib import Path


if __name__ == "__main__":

    '''
    exp_root = Path("./_exp")
    yolo_model_path = exp_root / "yolo11n.pt"

    # Load a model
    # model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO(yolo_model_path)  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n.yaml").load(yolo_model_path)  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="yolo_class_ab.yaml", project="./_exp/yolo_models", name="train", epochs=30, imgsz=640)
    '''

    # Load a pretrained YOLO11n model
    exp_root = Path("./_exp")
    yolo_model_path = exp_root / "yolo11n.pt"

    # model = YOLO(yolo_model_path)
    model = YOLO("./_exp/yolo_models/train4/weights/best.pt")

    # Run inference on 'bus.jpg' with arguments
    # model.predict("./_exp/test_yolo_frms", save=True, project="./_exp/yolo_results", name="predict", imgsz=320, conf=0.5)
    model.predict("./_exp/20240312_213521_seg1.mov", save=True, project="./_exp/yolo_results", name="predict", batch=16, imgsz=320, conf=0.5)