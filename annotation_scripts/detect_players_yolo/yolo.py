from ultralytics import YOLO

# load a model
model = YOLO("../../_exp/yolo11n-pose.pt")

# predict with model
project = "../../_exp/yolo_run"
results = model.predict(source="../../_exp/20240312_203520.MOV", show=False, save=True, project=project, name="predict")