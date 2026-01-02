from pathlib import Path

exp_root = Path("./_exp")
exp_ws_dir = exp_root / "ws_yolo"

yolo_cfg = {
    'exp_root': exp_root,
    'yolo_model_path': exp_ws_dir / "yolo11n.pt",  # path to pretrained YOLO model
    'yolo_custom_model_path': exp_ws_dir / "yolo_models" / "train" / "weights" / "best.pt",  # path to custom trained YOLO model

    'yolo_train_params': {
        'data': exp_root / "yolo_class_ab.yaml",  # path to data config yaml
        'project': exp_root / "yolo_models",  # path to save trained model
        'name': "train",  # name of training run
        'epochs': 30,  # number of training epochs
        'imgsz': 640,  # training image size}
    }
}