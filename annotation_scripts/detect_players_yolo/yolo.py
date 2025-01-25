from ultralytics import YOLO

# load a model
model = YOLO("yolo11n-pose.pt")

# predict with model
results = model.predict(source="20240312_203520.MOV", show=False, save=True)