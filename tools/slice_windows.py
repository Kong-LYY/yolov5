import os
import cv2
import time
import numpy as np 

import torch


def slice_single_image(img, sliceHeight=640, sliceWidth=640, overlap=0.2):

    win_h, win_w = img.shape[:2]

    # if slice sizes are large than image, pad the edges
    if sliceHeight > win_h:
        pad = sliceHeight - win_h
    if sliceWidth > win_w:
        pad = max(pad, sliceWidth - win_w)
    
    n_ims = 0
    dx = int((1. - overlap) * sliceWidth)   # 153
    dy = int((1. - overlap) * sliceHeight)

    point_x = []
    point_y = []

    for y0 in range(0, img.shape[0], dy):
        for x0 in range(0, img.shape[1], dx):
            n_ims += 1
            #
            #这一步确保了不会出现比要切的图像小的图，其实是通过调整最后的overlop来实现的
            #举例:h=6000,w=8192.若使用640来切图,overlop:0.2*640=128,间隔就为512.所以小图的左上角坐标的纵坐标y0依次为:
            #:0,512,1024,....,5120,接下来并非为5632,因为5632+640>6000,所以y0=6000-640
            if y0 + sliceHeight > img.shape[0]:
                y = img.shape[0] - sliceHeight
            else:
                y = y0
            if x0 + sliceWidth > img.shape[1]:
                x = img.shape[1] - sliceWidth
            else:
                x = x0


            point_x.append(x)
            point_y.append(y)
            # extract image
            slice_img = img[y:y + sliceHeight, x:x + sliceWidth]
    
    
    
# 滑窗并保存
def slice_single_image(image_path, save_name, outpath, is_save_imgs, sliceHeight=640, sliceWidth=640, overlap=0.2):

    image0 = cv2.imread(image_path, 1)  # color
    # ext = '.' + image_path.split('.')[-1]
    ext = '.jpg'
    win_h, win_w = image0.shape[:2]

    # if slice sizes are large than image, pad the edges
    # 避免出现切图的大小比原图还大的情况
    pad = 0
    if sliceHeight > win_h:
        pad = sliceHeight - win_h
    if sliceWidth > win_w:
        pad = max(pad, sliceWidth - win_w)
    
    # pad the edge of the image with black pixels
    if pad > 0:
        border_color = (0, 0, 0)
        image0 = cv2.copyMakeBorder(image0, pad, pad, pad, pad,
                                    cv2.BORDER_CONSTANT, value=border_color)

    n_ims = 0
    dx = int((1. - overlap) * sliceWidth)   # 153
    dy = int((1. - overlap) * sliceHeight)

    point_x = []
    point_y = []

    for y0 in range(0, image0.shape[0], dy):
        for x0 in range(0, image0.shape[1], dx):
            n_ims += 1
            #
            #这一步确保了不会出现比要切的图像小的图，其实是通过调整最后的overlop来实现的
            #举例:h=6000,w=8192.若使用640来切图,overlop:0.2*640=128,间隔就为512.所以小图的左上角坐标的纵坐标y0依次为:
            #:0,512,1024,....,5120,接下来并非为5632,因为5632+640>6000,所以y0=6000-640
            if y0 + sliceHeight > image0.shape[0]:
                y = image0.shape[0] - sliceHeight
            else:
                y = y0
            if x0 + sliceWidth > image0.shape[1]:
                x = image0.shape[1] - sliceWidth
            else:
                x = x0


            point_x.append(x)
            point_y.append(y)
            # extract image
            window_c = image0[y:y + sliceHeight, x:x + sliceWidth]

            if is_save_imgs:
                outname= os.path.join(outpath, save_name + \
                                                        '|' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(sliceWidth) + \
                                                        '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h) + ext)
                cv2.imwrite(outname, window_c)

# 保存滑窗左上角像素位置    
def save_slice_point():
    pass # 保证程序完整性


if __name__ == "__main__":
    not_use_multiprocessing = True
    is_save_imgs = True
    raw_images_dir  = '/data/2D-detection/yolov5-6.2/data/2022-11-10-test_image'   # 这里就是原始的图片

    save_image_path = '/data/2D-detection/yolov5-6.2/data/11-10-slice_result'

    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)

    img_names = os.listdir(raw_images_dir)

    for img in img_names :
        img_path = os.path.join(raw_images_dir, img)
        save_name = img.split('.')[0]
        slice_single_image(img_path, save_name, save_image_path, is_save_imgs,sliceHeight=2048, sliceWidth=2048, overlap=0.1)