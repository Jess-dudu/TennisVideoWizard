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

Address this video editing task with ML classification model. First, train a ML classification model to classify each frame to be in-point or not. Then, for each video to edit, process it in following steps:

1. Extract frames from input video at a given FPS (e.g., 5 frames/sec)
2. Run model on each frame to classify. 
3. Extract all the points that have streaks longer than some time threshold (points cannot be too short).

## Transfer Learning based ML Model

Use pretrained Resnet model and fine tune the last layer to do classification with following script: resnet50, 3 classes, 2 epoch, 1 gpu, 

python ./src/frame_classifier/resnet_classifier.py 50 3 2 ./_exp/Dataset/train ./_exp/Dataset/val -ts ./_exp/Dataset/test --save_path ./_exp/models --gpus 1 --transfer --tune_fc_only
