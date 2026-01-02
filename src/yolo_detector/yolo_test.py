from ultralytics import YOLO
from pathlib import Path

import yolo_detector.config as yolo_cfg

exp_root = yolo_cfg.yolo_cfg['exp_root']

def run_yolo_test(source_path):
    # weights of pretrained model
    yolo_model_path = exp_root / "yolo11n.pt"   # "yolo11n-pose.pt"
    # weights of trained model
    yolo_model_path = exp_root / "yolo_models" / "train" / "weights" / "best.pt"

    model = YOLO(yolo_model_path)

    # Run inference on 'bus.jpg' with arguments
    project_path = exp_root / "yolo_results"
    model.predict(source_path, save=True, project=project_path, name="predict", batch=16, imgsz=640, conf=0.5)

if __name__ == "__main__":
    source_path_0 = exp_root / "AO2017_cropped.jpg"
    source_path_1 = exp_root / "test_yolo_frms"
    source_path_2 = exp_root / "20240312_213521_seg1.mov"
    run_yolo_test(source_path_2)

