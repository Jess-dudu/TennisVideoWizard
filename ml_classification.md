## Classification Methods

Tennis video editing can be addressed with classification model, assuming the model can classify video frames (say every 1/10 of a second) to active-rally or between-point. Non-action frames (Between-point) can be automatically cut out then.

To train for mutually exclusive classes, ImageFolder (one subfolder for each class type and images of that class are stored under it) is used to refine a pre-trained model (e.g., Resnet18/50/101). It is first tested on Kaggle's Cats/Dogs/Horses Dataset (three classes). Then applied on my own tennis Dataset (two classes).

### Test with Kaggle's Cats/Dogs/Horses Dataset

A ML classification model based on pre-trained Resnet model is tested for multi-class animal classification task based on Kaggle's animal dataset (https://www.kaggle.com/datasets/arifmia/animal), which contains three type of animals (i.e., Cats/Dogs/Horses). After downloading the dataset, the model can be trained with following command:

python src/frame_classifier/train.py --config animals_cls3.yaml

Classification result seems quite good with just 10 epoch (image resized to 224 x 224):
- resnet18: test_acc_epoch = 0.9497206807136536
- resnet50: test_acc_epoch = 0.9916201233863831

Refer to src/frame_classifier/test_animals_model.ipynb for confusion matrix and study of resnet50 errors (3/358 errors, one due to dirty label, two due to images with both cat and dog).

Since some images have multiple animals, the dataset may be better handled with a multi-label classification model rather than multi-class model.

### Classify video frames to active/between points (2-class)

The resnet model is further trained to do the frame classification task. Many frames are extract from recorded tennis matches and separate to active/between points folders for training (train: 5286 images, test: 1791 images). The model can be trained with following command:

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

