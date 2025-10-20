# TennisVideoWizard
Try using deep learning models to analyze tennis match videos and cut out the non-action parts (between points)

## Classification Method

Previous task may be addressed with ML classification models. We can train a ML classification model to classify each frame to be active or between points (mutually exclusive). Then, cut out all the non-action frames from a given video. 

### Test with Kaggle's Cats/Dogs/Horses Dataset

A ML classification model based on pre-trained Resnet model is tested for multi-class animal classification task based on Kaggle's animal dataset (https://www.kaggle.com/datasets/arifmia/animal), which contains three type of animals (i.e., Cats/Dogs/Horses). After downloading the dataset, the model can be trained with following command:

python src/frame_classifier/train.py --config animals_cls3.yaml

Classification result seems quite good with just 10 epoch (image resized to 224 x 224):
- resnet18: test_acc_epoch = 0.9497206807136536
- resnet50: test_acc_epoch = 0.9916201233863831

Refer to src/frame_classifier/test_animals_model.ipynb for confusion matrix and study of resnet50 errors (3/358 errors, one due to dirty label, two due to images with both cat and dog).

Since some images have multiple animals, the dataset may be better handled with a multi-label classification model rather than multi-class model.

### Classify video frames to active/between points (2-class)

The resnet model is further trained to do the frame classification task. Many frames are extract from recorded tennis double matches and separate to active/between points folders for training (train: 5286 images, test: 1791 images). The model can be trained with following command:

python src/frame_classifier/train.py --config cls2_tennis.yaml

Initial classification result seems much worse than previous animal classification. Given that the test set has 562 (active frame) vs. 1229 (between points), always guessing "between points" can get 68.6% accuracy.
- resnet50 (tune_fc_only):  test_acc_epoch = 0.6917923092842102
- resnet50 (transfer only): test_acc_epoch = 0.7442769408226013
- resnet101 (grayscale): acc = 87% 

Reduce lr to 0.0001 & crop image to tighter frame & Grayscale input & RandomHorizontalFlip & Resnet101 (acc = 87%):

Confusion Matrix: 0-active (562), 1-between points (1229):
%83 (grayscale, Resnet50, epoch=10)
tensor([[ 424,  138],
        [ 182, 1047]])

%87 (grayscale, Resnet101, epoch=10)
tensor([[ 490,   72],
        [ 181, 1048]])

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

