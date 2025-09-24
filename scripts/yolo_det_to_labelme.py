'''
Author: dpsfigo
Date: 2025-03-03 14:55:11
LastEditors: dpsfigo
LastEditTime: 2025-03-03 16:54:57
Description: 请填写简介
'''
import sys
import os
current_file = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file))
sys.path.append(parent_directory)
import json
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import base64
from io import BytesIO

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def yolov8_det_to_labelme(results, image_path):
    # 加载图像
    image = Image.open(image_path)
    image_width, image_height = image.size

    # 将图像转换为base64编码的字符串
    image_data = image_to_base64(image)

    # 初始化LabelMe标注字典
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.split("/")[-1],
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width
    }


    # 遍历每个检测结果
    for result in results:
        # masks = result.masks.xy
        boxes = result.boxes
        names = result.names

        if boxes is not None:
            for i in range(len(boxes)):
                # 获取类别名称
                class_id = int(boxes.cls[i].item())
                class_name = names[class_id]

                # 获取分割掩码的多边形点
                point = boxes[i].xyxy.tolist()[0]
                point = [[point[0], point[1]], [point[2], point[3]]]
                # contours = []
                # for contour in cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                #     contour = contour.flatten().tolist()
                #     if len(contour) >= 6:  # 至少需要3个点来形成一个多边形
                #         contours.append(contour)

                # 为每个多边形添加标注信息
                shape = {
                    "label": class_name,
                    "points": point,
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                labelme_data["shapes"].append(shape)

    return labelme_data


if __name__ == "__main__":
    # 加载预训练的分割模型
    model = YOLO('/Users/qingyuan/cv/ultralytics/best_zhili.pt')

    # 定义图像路径
    img_path = "/Users/qingyuan/cv/qykw/zhili/test"
    imgs = os.listdir(img_path)
    for img in imgs:
        image_path = os.path.join(img_path, img)
        if not image_path.endswith('.jpg'):
            continue
        # image_path = '/Users/qingyuan/cv/datasets/coco8/images/train/000000000009.jpg'

        # 进行推理
        results = model(image_path)
        # result.save(filename=des_name,labels=False,line_width=2)  # save to disk
        # for result in results:
        #     result.save(filename="seg_result.jpg")  

        # 将YOLOv8分割结果转换为LabelMe格式
        labelme_data = yolov8_det_to_labelme(results, image_path)

        # 定义保存的JSON文件路径
        json_path = image_path.split(".")[0] + '.json'

        # 保存为LabelMe的JSON文件
        with open(json_path, 'w') as f:
            json.dump(labelme_data, f, indent=2)
    # image_path = '/Users/qingyuan/cv/datasets/coco8/images/train/000000000009.jpg'

    # 进行推理
    # results = model(image_path)
    # # result.save(filename=des_name,labels=False,line_width=2)  # save to disk
    # # for result in results:
    # #     result.save(filename="seg_result.jpg")  

    # # 将YOLOv8分割结果转换为LabelMe格式
    # labelme_data = yolov8_det_to_labelme(results, image_path)

    # # 定义保存的JSON文件路径
    # json_path = 'labelme_test.json'

    # # 保存为LabelMe的JSON文件
    # with open(json_path, 'w') as f:
    #     json.dump(labelme_data, f, indent=2)
