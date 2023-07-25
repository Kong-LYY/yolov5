
import xml.etree.ElementTree as ET
 
import pickle
import os
from os import listdir , getcwd
from os.path import join
import glob
import cv2 as cv
import numpy as np
import json
 
classes = ["Target",]
 
def convert(id, size, box):
 
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [id,x,y,w,h]

def write_line_value(fw, value):
    fw.write(str(value[0]))
    fw.write(' ')
    str_value = np.round(value[1:], 4)
    # fw.write(str_value)
    for single_value in str_value:
        fw.write(str(single_value))
        fw.write(' ')
    fw.write('\n') # 换行
 
def convert_annotation(image_name, xml_name, txt_save_name, img_save_name):

    # # 读图转化成jpg
    # img = cv.imread(image_name)
    # cv.imwrite(img_save_name, img)

    with open(xml_name) as f:
         root = json.load(f)

    # # label转化
    # f = open(xml_name)
    # xml_text = f.read()
    # root = ET.fromstring(xml_text)
    # f.close()
    try:
        result = root.get('step_1')
        bbox = result.get('result')
    except:
        fw = open(txt_save_name, 'w')
        fw.close()
        return




    img_w = int(root.get('width'))
    img_h = int(root.get('height'))

    if len(bbox) == 0:
        return
    # 读图转化成jpg
    img = cv.imread(image_name)
    cv.imwrite(img_save_name, img)
    fw = open(txt_save_name, 'w')
    for i in range(0, len(bbox)):


        b_w = bbox[0].get('width')
        b_h = bbox[0].get('height')

        b_x = bbox[0].get('x')
        b_y = bbox[0].get('y')

        bb = [0, (b_x+0.5*b_w)/img_w  ,(b_y+ 0.5*b_h)/img_h ,b_w/img_w ,b_h/img_h]
        write_line_value(fw, bb)
    fw.close()
def get_txt_value(name):
    # 获取txt值
    f = open(name)
    result = f.read().splitlines()
    f.close
    return result

def convert_dataset(names, img_path, json_path, save_path_img, save_path_label):

    for name in names:
        s_name = name.split('.')
        txt_save_name = os.path.join(save_path_label, s_name[0] + '.txt')
        img_save_name = os.path.join(save_path_img, s_name[0] + '.jpg')
        img_name = os.path.join(img_path, name)
        xml_name = os.path.join(json_path, name + '.json')
        convert_annotation(img_name, xml_name, txt_save_name, img_save_name)
def mk_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def parse_coco():
    root = '/mnt/d/yolov5/lack/crop_real'


    # 读取json并解析
    json_path = os.path.join(root, 'label')
    img_path  = os.path.join(root, 'image')

    save_img_path   = os.path.join(root, 'result_glass/images')
    save_label_path = os.path.join(root, 'result_glass/labels')
    mk_dir(save_img_path)
    mk_dir(save_label_path)
   
    save_path_img_train = os.path.join(save_img_path, 'train')
    save_path_img_val = os.path.join(save_img_path, 'val')
    save_path_img_test = os.path.join(save_img_path, 'test')
    mk_dir(save_path_img_train)
    mk_dir(save_path_img_val)
    mk_dir(save_path_img_test)

    save_path_label_train = os.path.join(save_label_path, 'train')
    save_path_label_val = os.path.join(save_label_path, 'val')
    save_path_label_test = os.path.join(save_label_path, 'test')
    mk_dir(save_path_label_train)
    mk_dir(save_path_label_val)
    mk_dir(save_path_label_test)


    train_names = os.listdir(img_path)

    convert_dataset(train_names, img_path, json_path, save_path_img_train, save_path_label_train)
    # convert_dataset(val_names, img_path, json_path, save_path_img_val, save_path_label_val)
    # convert_dataset(test_names, img_path, json_path, save_path_img_test, save_path_label_test)

 
if __name__ == '__main__':

    parse_coco()
    