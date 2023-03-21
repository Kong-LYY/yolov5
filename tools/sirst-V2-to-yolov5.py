
import xml.etree.ElementTree as ET
 
import pickle
import os
from os import listdir , getcwd
from os.path import join
import glob
import cv2 as cv
import numpy as np
 
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

    # 读图转化成jpg
    img = cv.imread(image_name)
    cv.imwrite(img_save_name, img)

    # label转化
    f = open(xml_name)
    xml_text = f.read()
    root = ET.fromstring(xml_text)
    f.close()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)


    fw = open(txt_save_name, 'w')
 
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            print(cls)
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        if xmlbox is None:
            continue
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert(cls_id, (w,h), b)
        write_line_value(fw, bb)
    fw.close()
def get_txt_value(name):
    # 获取txt值
    f = open(name)
    result = f.read().splitlines()
    f.close
    return result

def convert_dataset(names, img_path, xml_path, save_path_img, save_path_label):

    for name in names:
        txt_save_name = os.path.join(save_path_label, name + '.txt')
        img_save_name = os.path.join(save_path_img, name + '.jpg')
        img_name = os.path.join(img_path, name + '.png')
        xml_name = os.path.join(xml_path, name + '.xml')
        convert_annotation(img_name, xml_name, txt_save_name, img_save_name)
def mk_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def parse_sirst():
    root = '/data/2D-detection/yolov5/'
    # 获取训练集，验证集，测试集名称
    splits_path = root + 'dataset/open-sirst-v2/splits'
    train_names = get_txt_value(os.path.join(splits_path, 'train_full.txt'))
    val_names   = get_txt_value(os.path.join(splits_path, 'val_full.txt'))
    test_names  = get_txt_value(os.path.join(splits_path, 'test_full.txt'))

    # 读取XML并解析
    xml_path = root + 'dataset/open-sirst-v2/annotations/bboxes'
    img_path = root + 'dataset/open-sirst-v2/mixed'

    save_img_path = root + 'dataset/sirst-v2-yolov5-dataset/images'
    save_label_path = root + 'dataset/sirst-v2-yolov5-dataset/labels'
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

    convert_dataset(train_names, img_path, xml_path, save_path_img_train, save_path_label_train)
    convert_dataset(val_names, img_path, xml_path, save_path_img_val, save_path_label_val)
    convert_dataset(test_names, img_path, xml_path, save_path_img_test, save_path_label_test)

 
if __name__ == '__main__':

    parse_sirst()
    