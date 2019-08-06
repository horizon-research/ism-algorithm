fromm __future__ import print_function
import argparse
import os
import random
import skimage.io
import skimage.transform
import torch
from numpy import *
import numpy as np
import time
import math
import re
import sys
import cv2 as cv

# parse the arguments
parser = argparse.ArgumentParser(description='ISM_aglorithm')
parser.add_argument('--datapath', default=None,
                            help='select model with no "/" at front')
parser.add_argument('--loadmodel', default=None,
                            help='loading model')
parser.add_argument('--use-cuda', action='store_false', default=False,
                            help='enables CUDA')
parser.add_argument('--saveimg', type=bool, default=False,
                            help='save processed images')
parser.add_argument('--datasize', type=int, default=100,
                            help='data size of the imgs')
parser.add_argument('--no_pw', type=bool, default=False,
                            help='Dont apply propagation window')
parser.add_argument('--pw', type=int, default=4,
                            help='propagation window')
parser.add_argument('--p_size', type=int, default=7,
                            help='patch size')
args = parser.parse_args()

model = None
left_prefix = None
right_prefix = None
result_prefix = None
pw = None

def load_dnn_model(model_path):
    raise Exception('This function is not implemented.')

def check_setup():
    global model, pw, left_prefix, right_prefix, result_prefix
    if args.loadmodel == None:
        raise Exception('Model is not available.')

    model = load_dnn_model(args.loadmodel)

    if args.datapath == None:
        raise Exception('Data path is not available.')

    left_prefix = args.datapath + 'left/'
    right_prefix = args.datapath + 'right/'
    result_prefix = args.datapath + 'disparity/'

    if args.no_pw:
        pw = 1
    else:
        pw = args.pw

def load_disparity(inx)
    left_o = (skimage.io.imread(left_prefix + str(inx).zfill(4) + ".png")
                .astype('float32'))
    right_o = (skimage.io.imread(right_prefix + str(inx).zfill(4) + ".png")
                .astype('float32'))
    img_l = processed(left_o).numpy()
    img_r = processed(right_o).numpy()
    img_l = np.reshape(img_l,[1,3,img_l.shape[1],img_l.shape[2]])
    img_r = np.reshape(img_r,[1,3,img_r.shape[1],img_r.shape[2]])

    # padding zeros to the image edges
    top_pad = 544 - img_l.shape[2]
    left_pad = 960 - img_l.shape[3]
    img_l = np.lib.pad(img_l,((0,0),(0,0),(top_pad,0),(0,left_pad)),
                      mode='constant',constant_values=0)
    img_r = np.lib.pad(img_r,((0,0),(0,0),(top_pad,0),(0,left_pad)),
                      mode='constant',constant_values=0)

    disp = dnn_inference(img_l, img_r)
    return (img_l, img_r, disp)

def generate_opti_flow_imgs(inx, ii):
    # old grey image
    oldL_grey = cv.imread(left_prefix + str(inx).zfill(4) + ".png", 0)
    oldR_grey = cv.imread(right_prefix + str(inx).zfill(4) + ".png", 0)

    # curr image
    imgL_grey = cv.imread(left_prefix + str(inx+ii).zfill(4) + ".png", 0)
    imgR_grey = cv.imread(right_prefix + str(inx+ii).zfill(4) + ".png", 0)

    # optical flow result
    flow_l = cv.calcOpticalFlowFarneback(oldL_grey,imgL_grey, None,
                                        0.5, 4, 16, 5, 5, 1.2, 0)
    flow_r = cv.calcOpticalFlowFarneback(oldR_grey,imgR_grey, None,
                                        0.5, 4, 16, 5, 5, 1.2, 0)
    return (flow_l, flow_r)

def motion_compensation(esit_disp):
    for i in range(len(flowL)):
        for j in range(len(flowL[0])):
            # update the flow new indexes;
            flowL[i][j][0] = int(flowL[i][j][0])
            flowL[i][j][1] = int(flowL[i][j][1])
            flowR[i][j][0] = int(flowR[i][j][0])
            flowR[i][j][1] = int(flowR[i][j][1])
  
            # check if predict
            if prediction:
                xr = int(j-old_disp[i][j])
                yl = int(min(len(flowL)-1, max(0, i+flowL[i][j][1])))
                yr = int(min(len(flowR)-1, max(0, i+flowR[i][j][1])))
                y_m = int((yl+yr)/2)
                new_x = int(j+flowL[i][j][0])
    
                if (new_x >= 0 and new_x < len(flowL[0]) 
                    and xr >= 0 and xr < len(flowL[0])):
                    # update values;
                    esti_disp[y_m][new_x] = old_disp[i][j] + \
                                            flowL[i][j][0] - \
                                            flowR[i][xr][0]
                    esti_disp[y_m][new_x] = max(0, esti_disp[y_m][new_x])
    
    return esti_disp

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. 
'''
def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    endian = None
    
    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    
    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian
    
    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)
    return np.reshape(data, shape), scale
    

'''
Comparison function: use for images
'''
def compare_imgs(img, gt, bar, name):
    # compare the difference
    diff = np.absolute(np.subtract(img, gt))
    max_v = np.amax(diff)
    min_v = np.amin(diff)
    mean_v = np.mean(diff)
    cnt = 0.0
    # start to loop and calculate the err.
    for i in range(len(img)):
        for j in range(len(img[0])):
            diff = abs(img[i][j] - gt[i][j])
            if (diff > bar):
                cnt += 1
                area = img.shape[0]*img.shape[1]
                err_v = cnt/area
    
    print(name+"_comparison: ", max_v, min_v, mean_v, err_v)
    return mean_v, err_v

def main():
    check_setup()

    collected_result = {
        'pred_err' : [],
        'pw_err' : [],
    }

    # set prevous predicted disparity map to None
    old_disp = None
    old_l = None
    old_r = None

    for inx in range(1, data_size - pw, pw):
        if old_disp = None:
            (old_l, old_r, old_disp) = dnn_inference(inx)

        for ii in range(1, pw+1):
            (img_l, img_r, disp) = dnn_inference(inx, ii)
            
            (flow_l, flow_r) = generate_opti_flow_imgs(inx, ii)

            # copy the disp and proceed predict part
            esti_disp = old_disp.copy()
            esti_disp = motion_compensation(esti_disp)

	    # open the disparity result
            file = open(result_prefix + str(inx+ii).zfill(4) + ".pfm", "r")
	    res_disp, _ = load_pfm(file)
	    res_disp = np.flip(res_disp, 0)
		
	    # analysis
	    mean_v, err_v = compare_imgs(esti_disp, res_disp, bar, "esti")
	    collected_result["pw_err"].append(mean_v)
	    collected_result["pw_rate"].append(err_v)
	    mean_v, err_v = compare_imgs(pred_disp, res_disp, bar, "pred")
	    collected_result["pred_err"].append(mean_v)
	    collected_result["pred_rate"].append(err_v)
		
        # copy the previous pred_disp for next prediction
        old_disp = pred_disp.copy()
            

if __name == '__main__':
    main()
