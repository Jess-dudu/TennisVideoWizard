from ultralytics import YOLO
from pathlib import Path


if __name__ == "__main__":

    exp_root = Path("./_exp")
    yolo_model_path = exp_root / "yolo11n.pt"

    # Load a model
    # model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO(yolo_model_path)  # load a pretrained model (recommended for training)
    # model = YOLO("yolo11n.yaml").load(yolo_model_path)  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="yolo_class_ab.yaml", project="./_exp/yolo_models", name="train", epochs=30, imgsz=640)