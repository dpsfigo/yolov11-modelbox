# -*- coding: utf-8 -*-
import json
import os
import argparse
from tqdm import tqdm
import glob
import cv2
import numpy as np
import shutil


def create_save_dir(save_dir):
        images_dir = os.path.join(save_dir, 'images')
        labels_dir = os.path.join(save_dir, 'labels')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.mkdir(images_dir)
            os.mkdir(labels_dir)
        else:
            if not os.path.exists(images_dir):
                os.mkdir(images_dir)
            if not os.path.exists(labels_dir):
                os.mkdir(labels_dir)
        return images_dir + os.sep, labels_dir + os.sep

def convert_label_json(json_dir, save_dir, classes):
    json_paths = os.listdir(json_dir)
    classes = classes.split(',')

    for json_path in tqdm(json_paths):
        if not json_path.endswith('.json'):
            continue
        # for json_path in json_paths:
        path = os.path.join(json_dir, json_path)
        # print(path)
        with open(path, 'r', encoding='utf-8') as load_f:
            print(load_f)
            #json_dict = json.load(load_f, encoding='utf-8')
            json_dict = json.load(load_f)
        h, w = json_dict['imageHeight'], json_dict['imageWidth']

        # save txt path
        # create_save_dir(save_dir)
        txt_path = os.path.join(save_dir, json_path.replace('json', 'txt'))
        img_path = os.path.join(save_dir, json_path.replace('json', 'jpg'))
        shutil.copy(path[:-4]+"jpg", img_path)
        txt_file = open(txt_path, 'w')

        for shape_dict in json_dict['shapes']:
            label = shape_dict['label']
            label_index = classes.index(label)
            points = shape_dict['points']

            points_nor_list = []

            for point in points:
                points_nor_list.append(point[0] / w)
                points_nor_list.append(point[1] / h)

            points_nor_list = list(map(lambda x: str(x), points_nor_list))
            points_nor_str = ' '.join(points_nor_list)

            label_str = str(label_index) + ' ' + points_nor_str + '\n'
            txt_file.writelines(label_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='json convert to txt params')
    parser.add_argument('--json-dir', type=str, default='/home/yf/cv/ultralytics/data/zhili', help='json path dir')
    parser.add_argument('--save-dir', type=str, default='/home/yf/cv/ultralytics/datasets/zhili-seg', help='txt save dir')
    parser.add_argument('--classes', type=str, default='1_5_centrifuge_tube,50_centrifuge_tube,1000_tip', help='classes') # 设置标签名
    args = parser.parse_args()
    json_dir = args.json_dir
    save_dir = args.save_dir
    classes = args.classes
    convert_label_json(json_dir, save_dir, classes)


