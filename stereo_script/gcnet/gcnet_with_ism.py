#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import tempfile
from math import ceil
import cv2 as cv
import time
import re

import skimage
import skimage.io
import skimage.transform


parser = argparse.ArgumentParser()
parser.add_argument('--path', help='image path')
parser.add_argument('--datasize', type=int, default=100,
                    help='dataset size')
parser.add_argument('--ew',  help='extrapolate window', default=1, type=int)

args = parser.parse_args()

# set the threshold for the measure error
error_threshold = 3.0

select_idx = 32
draw_motion = False

# set extrapolate window size
ew = args.ew

'''
Load a PFM file into a Numpy array. Note that it will have
a shape of H x W, not W x H. Returns a tuple containing the
loaded image and the scale factor from the file.
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


def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,:]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close()

# comparison function
def compareDisp(index, flow):
    disp = flow[:,:,0:1]
    [height, width, channel] = blob.shape
    disp = np.absolute(np.reshape(disp, (height, width)))
    # gt = np.float32(cv.imread(args.path+"disp/"+str(index).zfill(4)+".png", 0))
    # this is to load PFM file
    file = open(args.path+"disparity/"+str(index).zfill(4) + ".pfm", "r")
    gt,_ = load_pfm(file)
    gt = np.flip(gt, 0)

    print(gt.shape, disp.shape)
    diff = np.abs(np.subtract(disp, gt))
    max_v = np.amax(diff)
    min_v = np.amin(diff)
    mean_v = np.mean(diff)

    print("[mean]",np.mean(diff))
    print("[max]", max_v)
    print("[min]", min_v)
    print(compareError(diff))
    return np.mean(diff)

'''
Comparison function: use for images
'''
def compare_imgs(img, gt):
    # compare the difference
    diff = np.absolute(np.subtract(img, gt))
    # print(diff)
    max_v = np.amax(diff)
    min_v = np.amin(diff)
    mean_v = np.mean(diff)
    print("## comparison ##: ", max_v, min_v, mean_v)
    return mean_v


# compute the error rate for the this frame
def compareError(diff):
    error_cnt = 0
    for arr in diff:
        for element in arr:
            if element > error_threshold:
                error_cnt += 1
    (height, width) = diff.shape
    return float(error_cnt)/(height*width)

compare_result = {
        "pred" : [],
        "esti" : [],
        }


# set up path
saved_dir = "result/"
left_prefix = args.datapath + "left/"
right_prefix = args.datapath + "right/"
disp_prefix = args.datapath + "disp/"


for index in range(1,args.datasize-ew,ew):
    os.system("./nvstereo resnet18_2D 513 257 trt_weights.bin " \
                + left_prefix + str(index).zfill(4) + ".png " \
                + right_prefix + str(index).zfill(4) + ".png" \
                + " ./disp")

    # read the result
    pred_disp =  np.asarray(cv.imread("disp.png", 0)).astype(float)
    res_disp = np.asarray(cv.imread(args.datapath+"disp/"+str(index).zfill(4)+".png", 0)).astype(float)

    # writeFlow(args.out, blob)
    res = compareDisp(pred_disp, res_disp)
    old_disp = pred_disp

    # use motion compensation to compute the consecutive frames within the ew
    # read old grey image
    oldL_grey = cv.imread(args.path+"left/"+str(index).zfill(4)+".png", 0)
    oldR_grey = cv.imread(args.path+"right/"+str(index).zfill(4)+".png", 0)

    for ii in range(1, ew+1):
        print("EW: %d" % (ii))
        # curr image
        imgL_grey = cv.imread(args.path+"left/"+str(index+ii).zfill(4)+".png", 0)
        imgR_grey = cv.imread(args.path+"right/"+str(index+ii).zfill(4)+".png", 0)

        # openCV return matrix of (x, y)
        flowL = cv.calcOpticalFlowFarneback(oldL_grey,imgL_grey, 0.5, 4, 16, 5, 5, 1.2, 0)
        flowR = cv.calcOpticalFlowFarneback(oldR_grey,imgR_grey, 0.5, 4, 16, 5, 5, 1.2, 0)

        # copy the disp and proceed predict part
        esti_disp = old_disp.copy()
        print(old_disp.shape, flowL.shape, flowR.shape)

        # this two are used to draw motions
        outL = cv.imread(args.path+"left/"+str(index+ii).zfill(4)+".png")

        for i in range(len(flowL)):
            for j in range(len(flowL[0])):
                # cast flow into ints;
                flowL[i][j][0] = int(flowL[i][j][0])
                flowL[i][j][1] = int(flowL[i][j][1])
                flowR[i][j][0] = int(flowR[i][j][0])
                flowR[i][j][1] = int(flowR[i][j][1])

                if draw_motion and i % select_idx == 0 and j % select_idx  == 0:
                    # draw line to the output;
                    xl = int(min(len(flowL[0])-1, max(0, j+flowL[i][j][0])))
                    yl = int(min(len(flowL)-1, max(0, i+flowL[i][j][1])))
                    xr = int(min(len(flowR[0])-1, max(0, j+flowR[i][j][0])))
                    yr = int(min(len(flowR)-1, max(0, i+flowR[i][j][1])))

                    # draw lines in the images;
                    cv.line(outL, (j, i), (int(j+old_disp[i][j][0]), i), (0,255,0), 1)
                    # print(old_disp[i][j])


                # motion compensate disparity
                xr = int(j-old_disp[i][j])
                yl = int(min(len(flowL)-1, max(0, i+flowL[i][j][1])))
                yr = int(min(len(flowR)-1, max(0, i+flowR[i][j][1])))
                y_m = int((yl+yr)/2)
                new_x = int(j+flowL[i][j][0])

                if new_x >= 0 and new_x < len(flowL[0]) and xr >= 0 and xr < len(flowL[0]):
                    # update values;
                    esti_disp[y_m][new_x] = old_disp[i][j] + flowL[i][j][0] - flowR[i][xr][0]
                    esti_disp[y_m][new_x] = max(0, esti_disp[y_m][new_x])

        # if want to check the motion
        if draw_motion:
            print("outL shape: ", outL.shape)
            skimage.io.imsave(str(index+ii).zfill(4)+"_left_flow.jpg", outL)
            # end of checking

        
        os.system("./nvstereo resnet18_2D 513 257 trt_weights.bin " \
                    + left_prefix + str(index).zfill(4) + ".png " \
                    + right_prefix + str(index).zfill(4) + ".png" \
                    + " ./disp")

        res_disp = np.asarray(cv.imread(args.datapath+"disp/"+str(index).zfill(4)+".png", 0)).astype(float)
        res = compare_imgs(esti_disp, res_disp)
        compare_result["esti"].append(res)

        pred_disp =  np.asarray(cv.imread("disp.png", 0)).astype(float)
        res = compare_imgs(esti_disp, res_disp)
        compare_result["pred"].append(res)


print("[STEREO] pred: ",args.path,np.mean(compare_result["pred"]))
print("[STEREO] esti: ",args.path,np.mean(compare_result["esti"]))
