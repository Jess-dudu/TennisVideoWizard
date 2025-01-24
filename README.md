# TennisVideoWizard
Try using deep learning to analyze tennis match videos and cut out the non-action parts (between points)

## Annotation

Start from manual annotation to understand the challenges in the task better.

1. Decide a json format to save annotation on video files (e.g., video file: short_clip1.mov, annotation file: short_clip1.mov.json)
2. Annotate on a short clip (play clip and write down the start & end timestamp of each point)
3. Write script (Use ffmpeg-python) to load json file and extract frames to verify annotation accuracy
4. To improve annotation accuracy, use VLC + Time v3.2 extension to display millisec while playing video

Some challenges observed:
- How to cut out the serves that stopped due to bad tosses (it needs to look into the future to decide)?
- Should we treat 1st serve fault as a seperate point or not? The gap between 1st and 2nd serve can vary a lot. May want to cut out the long ones.
- Under-hand serve may be hard to detect.
- End of point may be harder to pin point accurately.

## Classification Method

Address this video editing task with ML classification model. First, train a ML classification model to classify each frame to be active or between points. Then, for each video, process it in following steps:

1. Extract frames from input video at a given FPS (e.g., 5 frames/sec)
2. Run model on each frame to classify active/between points. 
3. Extract all the points that have streaks longer than some time threshold (points cannot be too short).

### Classify Kaggle's Cats/Dogs/Horses Dataset

Fine-tune pretrained Resnet model for Cats/Dogs/Horses dataset (https://www.kaggle.com/datasets/arifmia/animal):

python ./src/frame_classifier/train.py 50 3 5 ./_exp/Dataset/train ./_exp/Dataset/val -ts ./_exp/Dataset/test --save_path ./_exp/Dataset/models --gpus 1 --transfer --tune_fc_only

Classification result seems quite good with just 10 epoch (image resized to 200 x 200):
- resnet18: test_acc_epoch = 0.9497206807136536
- resnet50: test_acc_epoch = 0.994413435459137

Confusion Matrix: 0-cats (66), 1-dogs (250), 2-horses (42):
tensor([[ 65,   1,   0],
        [  0, 250,   0],
        [  1,   0,  41]])

### Classify video frames to active/between points (2-class)

Extract frames every 1/3 second from some tennis match videos and separate to active/between folders (train: 5286 images, validation: 1791 images). Then, train two-class classification model.

python ./src/frame_classifier/train.py 50 2 5 ./_exp/ClassAB/train ./_exp/ClassAB/val -ts ./_exp/ClassAB/test --save_path ./_exp/ClassAB/models --gpus 1 --transfer --tune_fc_only

Classification result seems very bad (test set has 562 vs. 1229 images, always guess "between points" can get 68.6% accuracy)
- resnet50 (tune_fc_only):  test_acc_epoch = 0.6917923092842102
- resnet50 (transfer only): test_acc_epoch = 0.7442769408226013
- resnet50 (num_classes=2): test_acc_epoch = 0.7264098525047302

Confusion Matrix: 0-active (562), 1-between points (1229):
tensor([[ 195,  367],
        [ 124, 1105]])

