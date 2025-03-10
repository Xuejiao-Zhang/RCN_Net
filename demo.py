# -*- coding: utf-8 -*-
"""
@ Time ： 2024/2/2 11:33
@ Auth ： xuejiaozhang
@ File ：demo.py
@ IDE ：PyCharm
@ Function :
"""

import cv2
import os
import torch
import math
from utils.rcn import RCN





if __name__ == '__main__':
    #path of model
    #model = 'path_to_pretrained_model'
    input_path = 'demo/input'
    output_path = 'demo/output'

    # device
    device_name = "cuda:1" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    # init
    rcn = RCN(model, device=device_name)
    with torch.no_grad():
        for file in os.listdir(input_path):

            #print('processing ', file)

            img_file = os.path.join(input_path, file)
            img = cv2.imread(img_file)

            grasps, x1, y1 = rcn.predict(img, device, mode='all', thresh=0.3, peak_dist=2)	# predict num
            im_rest = drawGrasps(img, grasps, mode='arrow')  # draw grasp box

            rect = [0, 0, 640, 480]
            drawRect(im_rest, rect)

            # save
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            save_file = os.path.join(output_path, file)
            cv2.imwrite(save_file, im_rest)

    print('FPS: ', rcn.fps())
