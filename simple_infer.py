import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (check_img_size, check_requirements, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        imgsz=(1280, 1280),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment

):
    source = str(source)
    # 创建路径
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    text_save = save_dir / 'labels'
    text_save.mkdir(parents=True, exist_ok=True) 
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)

    img_names = os.listdir(source)
    for img_name in img_names :
        img_path  = os.path.join(source, img_name)
        save_name = img_name.split('.')[0]
        txt_path = str(save_dir / 'labels' / save_name)  # im.txt
        
        # 单张小图推理
        crop_infer(img_path = img_path, model = model, txt_path = txt_path, conf_thres = conf_thres, iou_thres = iou_thres) 
            
# 滑窗裁剪
def get_slice_xy_img(img, y0, x0, sliceWidth, sliceHeight):
    if y0 + sliceHeight > img.shape[0]:
        y = img.shape[0] - sliceHeight
    else:
        y = y0
    if x0 + sliceWidth > img.shape[1]:
        x = img.shape[1] - sliceWidth
    else:
        x = x0
    # 裁剪
    slice_img = img[y:y + sliceHeight, x:x + sliceWidth]
    return x, y, slice_img

# 图像预处理
def img_process(img, model, sliceHeight, sliceWidth):

    im_infer = letterbox(img, (sliceHeight,sliceWidth), stride=model.stride, auto=model.pt)[0]  # padded resize
    img0 = img.copy()
    im_infer = im_infer.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im_infer = np.ascontiguousarray(im_infer)  # contiguous
    im_infer = torch.from_numpy(im_infer).to(model.device)
    im_infer = im_infer.float()
    
    im_infer /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im_infer.shape) == 3:
        im_infer = im_infer[None]  # expand for batch dim

    return im_infer, img0

def crop_infer(img_path, model, txt_path, conf_thres, iou_thres, sliceHeight=1280, sliceWidth=1280, overlap=0.2):
    
    img = cv2.imread(img_path) # 读单张图
    win_h, win_w = img.shape[:2] # 大图高宽

    dx = int((1. - overlap) * sliceWidth)  
    dy = int((1. - overlap) * sliceHeight)

    for y0 in range(0, img.shape[0], dy):
        for x0 in range(0, img.shape[1], dx):
            x, y, slice_img= get_slice_xy_img(img, y0, x0, sliceWidth, sliceHeight) # 获取裁剪起点x, y 
            im_infer, img0 = img_process(slice_img, model, sliceHeight, sliceWidth) # 图像预处理
            pred = model(im_infer) # 推理
            pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

            # 解析结果
            for i, det in enumerate(pred):
                if len(det):                    
                    gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    # 保存结果
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # 获取图片缺陷全局坐标
                        xywh[0] = (xywh[0] * sliceWidth  + x) / win_w
                        xywh[1] = (xywh[1] * sliceHeight + y) / win_h
                        xywh[2] = (xywh[2] * sliceWidth) / win_w
                        xywh[3] = (xywh[3] * sliceHeight) / win_h
                        line = (cls, *xywh, conf)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/1280_train_v5n6/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / '/data/2D-detection/yolov5/dataset/orin_data_overpaint/2020-10-28-11-06-SG3-GDS-20-0030', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
   
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)






    
    
    