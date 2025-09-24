'''
Author: dpsfigo
Date: 2025-09-11 11:44:44
LastEditors: dpsfigo
LastEditTime: 2025-09-11 13:37:01
Description: 请填写简介
'''
import os
import sys
current_file = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file))
sys.path.append(parent_directory)

from ultralytics import SAM

# Load the model
model = SAM("mobile_sam.pt")

# Predict a segment based on a single point prompt
results = model.predict("ultralytics/assets/zidane.jpg",conf=0.8)
output_path = "./output"

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    img_name = result.path.split("/")[-1]
    des_name = os.path.join(output_path,img_name)
    result.save(filename=des_name,labels=False,line_width=2)  # save to disk

