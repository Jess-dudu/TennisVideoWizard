## Player Detection Methods

YOLO11 is a very efficient object detection model and can detect players quite reliablly. It also can be retrained with customized training data. To generate training data effectively, some assumptions are made to simplify training data generation.

Since many frames are collected for active-point and between-point already, it is assumed that all players are active players if detected on active-point frames, relaxed players if detected on between-point frames. 

yolo_gen_label.py generates training data for Yolo model based on above assumptions. To run it, use "just yolo_annotate".

yolo_class_ab_train.py retrain Yolo11 to detect active/relaxed players. To run it, use "just yolo_train"




