# TennisVideoWizard

Self recorded tennis match videos usually consist too much down time and need to be edited out to save time and storage. Deep learning models might be trained to do that instead of manually done.

## Classification Methods

It is first tested with classification methods (classify each frame to active-point or between-point). A pre-trained network is fine-tuned to do classification. First tested on Kaggle's animal dataset (cats/dogs/horses) with 99% accuracy (three classes). 

Then it was applied to frame classification. It seems a lot tougher than animal classification. Resnet101 was used to achieve 87% accuracy (two classes) 

For details, please refer to [ml_classification.md](ml_classification.md).

## Player Detection Methods

Another way to address this may be to rely on object detection model, since players are usually in a more intense pose during active rallies. If active players are detected in given frame, that frame is most likely during active rally then.

Yolo11 is used to detect players in the videos. Assuming the players detected in active frames (annotated for classification methods) are active players. For between-point frames, the players are all relaxed players. 

The detection result can be used to retrain the Yolo model to detect active/relaxed players and then to decide each frame is active or not. For details, please refer to [ml_player_detection.md](ml_player_detection.md).

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

