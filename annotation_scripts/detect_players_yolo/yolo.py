from ultralytics import YOLO

# load a model
# model = YOLO("./_exp/yolo11n.pt")
model = YOLO("./_exp/yolo11n-pose.pt")
model.conf = 0.35

# predict with model
project = "./_exp/yolo_run"
results = model.predict(source="./_exp/AO2017_cropped.jpg", show=False, save=True, project=project, name="predict")