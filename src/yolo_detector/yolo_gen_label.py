'''
This script generates training data for active/between-point player detection for YOLO

imgs_class_path (_exp/ClassAB/train): 
    path to folder containing images (Active/Between) in class subfolders

train_gen_path (_exp/yolo_label): 
    path to folder where generated training data will be stored: images/labels subfolders
'''

from ultralytics import YOLO
from pathlib import Path
import shutil

''' Create training folder structure if not exists '''
def check_and_create_train_folder(train_gen_path):
        train_gen_path.mkdir(parents=True, exist_ok=True)
        (train_gen_path / "images").mkdir(parents=True, exist_ok=True)
        (train_gen_path / "labels").mkdir(parents=True, exist_ok=True)

''' Generate training data: copy image and create label file '''
def genTrainingData(img_path, boxes, train_gen_path, log_level=1):
    gen_img_path = train_gen_path / "images"
    gen_label_path = train_gen_path / "labels"

    # copy image to train images folder
    dest_img_path = gen_img_path / img_path.name
    shutil.copy(img_path, dest_img_path)

    # parent folder name: class name (Active/Between)
    label_fname = img_path.stem + ".txt"
    label_cls = 0 # target class for active players
    if str(img_path.parent.name).lower() == "between":
         label_cls = 1 # target class for between points
    if log_level > 1:
         print(label_fname, label_cls)

    with open(gen_label_path / label_fname, 'w') as file:
        person_cls = 0 # Yolo class for person
        for i in range(boxes.shape[0]):
            cls = boxes.cls[i]
            rec = boxes.xywhn[i]

            if cls != person_cls:
                if log_level > 0:
                     print(f"Skipping box {i} with class {cls}")
                continue
            
            if log_level > 0:
                 print(f"Box {i}: {label_cls}, {rec[0]}, {rec[1]}, {rec[2]}, {rec[3]}")
            
            file.write(f"{label_cls}  {rec[0]:.4f}  {rec[1]:.4f}  {rec[2]:.4f}  {rec[3]:.4f}\n")


if __name__ == "__main__":

    exp_root = Path("./_exp")
    yolo_model_path = exp_root / "yolo11n.pt"
    train_gen_path = exp_root / "yolo_label" / "train"

    # imgs_class_path = exp_root/ "ClassAB/test"
    imgs_class_path = exp_root/ "ClassAB/train"

    imgs_list = [x for x in imgs_class_path.glob("*/*.jpg") if x.is_file()] 
    # imgs_list = [imgs_class_path / "Active" / "00022.jpg"]
    # imgs_list = imgs_list[:10]
    print(len(imgs_list))

    # load a model
    model = YOLO(yolo_model_path)
    # model.conf = 0.35

    # predict with model
    # project = exp_root / "yolo_label_run"
    # model.predict(source=imgs_list, show=False, save=True, project=project, name="predict")

    batch_size = 10

    check_and_create_train_folder(train_gen_path)
    for i in range(0, len(imgs_list), batch_size):
        batch_imgs = imgs_list[i:i+batch_size]
        print(i, "/", len(imgs_list))

        results = model(source=batch_imgs)
        for idx, r in enumerate(results):
            boxes = r.boxes.cpu().numpy()
            genTrainingData(batch_imgs[idx], boxes, train_gen_path)
