import sys
from pathlib import Path

import cv2
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix()) 

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, is_ascii, non_max_suppression, scale_coords, set_logging
from utils.plots import Annotator


@torch.no_grad()
def run(weights='./runs/train/b32/weights/best.pt', source='./smoking scene_209.jpg',  
        imgsz=[640, 640],  conf_thres=0.25, iou_thres=0.45, max_det=1000,):

    save_dir = './make_video/'  

    set_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    w = weights[0] if isinstance(weights, list) else weights
    stride, names = 64, [f'class{i}' for i in range(1000)] 

    model = attempt_load(weights, map_location=device)  
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # class names
    
    imgsz = check_img_size(imgsz, s=stride)  
    ascii = is_ascii(names) 

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)

    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    for path, img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float() 
        img = img / 255.0  
        if len(img.shape) == 3:
            img = img[None]  

        pred = model(img, augment=False, visualize=False)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

        for i, det in enumerate(pred):  
            p, im0 = path, im0s.copy()
            p = Path(p)  
            save_path = save_dir + str(p.name)  
            pimg = im0.copy()

            #annotator = Annotator(im0, line_width=3, pil=not ascii)
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    #label = f'{names[int(cls)]} {conf:.2f}'
                    #annotator.box_label(xyxy, label, color=colors(int(cls), True))
                    #############################################################################
                    x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
                    pix_img = pimg[int(y1):int(y2),int(x1):int(x2)]
                    #print(pix_img.shape)
                    w = pix_img.shape[0]
                    h = pix_img.shape[1]
                    temp = cv2.resize(pix_img, (8,8), interpolation=cv2.INTER_LINEAR)
                    pix_img = cv2.resize(temp, (h,w), interpolation=cv2.INTER_NEAREST)
                    #print(pix_img.shape)
                    pimg[int(y1):int(y2),int(x1):int(x2)] = pix_img
                    cv2.imwrite("TEST.jpg", pimg)
                    #############################################################################

            #im0 = annotator.result()
            cv2.imwrite(save_path, pimg)


def main():
    run()

if __name__ == "__main__":
    main()