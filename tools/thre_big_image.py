
import os
import cv2 as cv

def save_threshold_result(values, save_path):

    a = 1

if __name__ == "__main__":
    #  读图

    image_path = '/data/2D-detection/yolov5-6.2/data/glass'

    image_names = os.listdir(image_path)

    for img_name in image_names:
        img_path = os.path.join(image_path, img_name)
        cv.imread(img_path, 1)

        # 大津法
        image=cv.cvtColor(image,cv.COLOR_RGB2GRAY)
        ret,binary=cv.threshold(image,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

        