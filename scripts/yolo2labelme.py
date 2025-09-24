'''
Author: dpsfigo
Date: 2025-01-20 14:40:13
LastEditors: dpsfigo
LastEditTime: 2025-01-20 15:12:25
Description: 请填写简介
'''
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 17:42:11 2022
@author: https://blog.csdn.net/suiyingy?type=blog
"""
import cv2
import os
import json
import shutil
import numpy as np
from pathlib import Path
import sys

current_file = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file))
sys.path.append(parent_directory)

from ultralytics import YOLO
 
id2cls = {0:'pt01', 1:'pt02', 2:'pt03', 3:'pt04', 4:'pt05', 5:'pt06', 6:'pt07', 7:'pt08', 8:'pt09', 9:'pt10', 10:'pt11', 11:'pt12', 12:'pt13'}
model = YOLO("yolo11n-pose.pt")
 
def xyxy2labelme(labels, w, h, image_path, save_dir='res/'):
    save_dir = str(Path(save_dir)) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    label_dict = {}
    label_dict['version'] = '5.0.1'
    label_dict['flags'] = {}
    label_dict['imageData'] = None
    label_dict['imagePath'] = image_path
    label_dict['imageHeight'] = h
    label_dict['imageWidth'] = w
    label_dict['shapes'] = []
    for l in range(len(labels[:13])):
        tmp = {}
        tmp['label'] = id2cls[l]
        tmp['points'] =[[float(labels[l][0]),float(labels[l][1])]]
        tmp['group_id']= None
        tmp['shape_type'] = 'point'
        tmp['flags'] = {}
        label_dict['shapes'].append(tmp)    
    fn = save_dir+image_path.rsplit('.', 1)[0]+'.json'
    with open(fn, 'w') as f:
        json.dump(label_dict, f)
 
def yolo2labelme(yolo_image_dir, save_dir='res/'):
    yolo_image_dir = str(Path(yolo_image_dir)) + '/'
    save_dir = str(Path(save_dir)) + '/'
    image_files = os.listdir(yolo_image_dir)
    for iimgf, imgf in enumerate(image_files):
        print(iimgf+1, '/', len(image_files), imgf)
        results = model(os.path.join(yolo_image_dir, imgf))
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        labels_xy = np.squeeze(results[0].keypoints.xy.cpu().numpy())[0]
        
        image = cv2.imread(yolo_image_dir + imgf)
        h,w = image.shape[:2]
        xyxy2labelme(labels_xy, w, h, imgf, save_dir)
    print('Completed!')
 
if __name__ == '__main__':
    image_dir = '/Users/qingyuan/cv/datasets/coco8-pose/images/train'
    save_dir = '/Users/qingyuan/Downloads/test_pos'
    yolo2labelme(image_dir, save_dir)
