'''
Author: dpsfigo
Date: 2025-01-04 10:34:00
LastEditors: dpsfigo
LastEditTime: 2025-01-04 10:36:09
Description: 请填写简介
'''
import os
import sys
current_file = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file))
sys.path.append(parent_directory)

from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("/Users/qingyuan/go/src/QingYuanServer-zhili/controllers/application/cvinfer/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
