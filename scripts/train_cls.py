'''
Author: dpsfigo
Date: 2024-11-23 13:20:22
LastEditors: dpsfigo
LastEditTime: 2024-12-20 19:22:17
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
model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="mnist160", epochs=100, imgsz=256, device="cpu")
