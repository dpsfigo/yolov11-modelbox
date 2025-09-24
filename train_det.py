'''
Author: dpsfigo
Date: 2024-11-23 13:34:24
LastEditors: dpsfigo
LastEditTime: 2025-09-19 14:16:07
Description: 请填写简介
'''
import os
import sys
current_file = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file))
sys.path.append(parent_directory)

from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
#model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

# Train the model
#results = model.train(data="coco8.yaml", epochs=1, imgsz=640, device="cpu")

model = YOLO("/Users/qingyuan/cv/ultralytics/ultralytics/cfg/models/11/yolo11-MobileNetV1.yaml")
model.load('yolo11n.pt') 
model.train(data='coco8.yaml', 
            imgsz=640,
            device='cpu',
            epochs=150,
            batch=4,
            optimizer='SGD',
            amp=False
            )