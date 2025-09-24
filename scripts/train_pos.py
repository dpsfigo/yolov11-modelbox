'''
Author: dpsfigo
Date: 2024-10-17 11:47:10
LastEditors: dpsfigo
LastEditTime: 2024-11-23 13:45:15
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
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-pose.yaml").load("yolo11n-pose.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8-pose.yaml", epochs=100, imgsz=640, device="cpu")
