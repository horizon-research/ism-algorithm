#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import cv2 as cv
import time
import re

import skimage
import skimage.io
import skimage.transform


parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
parser.add_argument('--path', help='image path')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
parser.add_argument('--ew',  help='extrapolate window', default=4, type=int)
parser.add_argument('--p_size',  help='patch window', default=5, type=int)
parser.add_argument('--data_size',  help='dataset size', default=50, type=int)
parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')

args = parser.parse_args()

if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)

# set the threshold for the measure error
error_threshold = 3.0

select_idx = 32
draw_motion = False

# set extrapolate window size
ew = args.ew
p_size = args.p_size
data_size = args.data_size


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
    file = open(args.path+"disp/"+str(index).zfill(4) + ".pfm", "r")
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


# compute the error rate for the this frame
def compareError(diff):
    error_cnt = 0
    for arr in diff:
        for element in arr:
            if element > error_threshold:
                error_cnt += 1
    (height, width) = diff.shape
    return float(error_cnt)/(height*width)

compare_result = []

for index in range(1,data_size-ew, 1):
    num_blobs = 2
    input_data = []
    # read two image data
    img0 = misc.imread(args.path+"right/"+str(index).zfill(4)+".png")
    print(img0.shape)
    if len(img0.shape) < 3:
        input_data.append(img0[np.newaxis, np.newaxis, :, :])
    else:
        input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

    img1 = misc.imread(args.path+"left/"+str(index).zfill(4)+".png")

    if len(img1.shape) < 3:
        input_data.append(img1[np.newaxis, np.newaxis, :, :])
    else:
        input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

    width = input_data[0].shape[3]
    height = input_data[0].shape[2]
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height

    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)

    proto = open(args.deployproto).readlines()

    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))

        tmp.write(line)

    tmp.flush()

    if not args.verbose:
        caffe.set_logging_disabled()
    caffe.set_device(args.gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)

    input_dict = {}
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

#
# There is some non-deterministic nan-bug in caffe
# it seems to be a race-condition
#
    print('Network forward pass using %s.' % args.caffemodel)
    i = 1
    while i<=5:
        i+=1
        net.forward(**input_dict)

        containsNaN = False
        for name in net.blobs:
            blob = net.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()

            if has_nan:
                print('blob %s contains nan' % name)
                containsNaN = True

        if not containsNaN:
            print('Succeeded.')
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')

    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
    # writeFlow(args.out, blob)
    res = compareDisp(index, blob)
    old_disp = (blob[:,:,0:1].astype('int'))

    # use motion compensation to compute the consecutive frames within the ew
    # read old grey image
    oldL_grey = cv.imread(args.path+"left/"+str(index).zfill(4)+".png", 0)
    oldR_grey = cv.imread(args.path+"right/"+str(index).zfill(4)+".png", 0)

    for ii in range(1, ew+1):
        print("EW: %d" % (ii))
        # curr image
        imgL_grey = cv.imread(args.path+"left/"+str(index+ii).zfill(4)+".png", 0)
        imgR_grey = cv.imread(args.path+"right/"+str(index+ii).zfill(4)+".png", 0)

        start_time = time.time()
        # openCV return matrix of (x, y)
        flowL = (cv.calcOpticalFlowFarneback(oldL_grey,imgL_grey, 0.5, 3, 15, 3, 5, 1.1, 0).astype('int'))
        flowR = (cv.calcOpticalFlowFarneback(oldR_grey,imgR_grey, 0.5, 3, 15, 3, 5, 1.1, 0).astype('int'))
        print('[optical] time = %.2f' %(time.time() - start_time))


        # copy the disp and proceed predict part
        esti_disp = (old_disp.copy()).astype('float32')
        print(old_disp.shape, flowL.shape, flowR.shape)

        # this two are used to draw motions
        outL = cv.imread(args.path+"left/"+str(index+ii).zfill(4)+".png")
        outR = cv.imread(args.path+"right/"+str(index+ii).zfill(4)+".png")

        start_time = time.time()
        len_flowL_y = len(flowL)
        len_flowL_x = len(flowL[0])
        for i in range(len_flowL_y):
            # t1 = time.time()
            for j in range(len_flowL_x):
                # motion compensate disparity
                xr = int(j-old_disp[i][j])
                yl = int(min(len_flowL_y-1, max(0, i+flowL[i][j][1])))
                yr = int(min(len_flowL_y-1, max(0, i+flowR[i][j][1])))
                y_m = int((yl+yr)/2)
                new_x = int(j+flowL[i][j][0])

                if new_x >= 0 and new_x < len_flowL_x and xr >= 0 and xr < len_flowL_y:
                    xl_patch = new_x
                    xr_patch = int(xr + flowL[i][j][0])
                    best_index = -(2*p_size+1)
                    best_patch = xr_patch
                    best_result = float('Inf')

                    xl_p = xl_patch
                    l_patch = outL[y_m : y_m+4, xl_p : xl_p + p_size]
                    for i_p in range(-2*p_size, 2*p_size):
                         xr_p = xr_patch + i_p
                         # if xr_p < 0 or xr_p + p_size >= len_flowL_x or \
                         #    xl_p < 0 or xl_p + p_size >= len_flowL_x \
                         #       or y_m < 0 or y_m + 4 >= len_flowL_y:
                         # extract the patches;
                         # print((y_m, y_m+4), (xr_p, xr_p+p_size), (xl_p, xl_p+p_size))
                         r_patch = outR[y_m : y_m+4, xr_p : xr_p + p_size]
                         # l_patch = outL[y_m : y_m+4, xl_p : xl_p + p_size]
                         # print(xl_p, xr_p, l_patch, r_patch)
                         if r_patch.shape != l_patch.shape:
                             continue

                         tmp_patch_result = np.sum(np.abs(np.subtract(l_patch, r_patch)))
                         if tmp_patch_result < best_result:
                             best_result = tmp_patch_result
                             best_index = i_p
                             best_patch = xr_patch + i_p
                         elif tmp_patch_result == best_result:
                             if abs(i_p) < best_index:
                                 best_index = i_p
                                 best_patch = xr_patch + i_p

                    # print(i, j, (xl_patch, xr_patch, (xl_patch-best_patch), old_disp[i][j]), best_index, best_result)

                    if best_result == float('Inf') or best_result > 200:
                        esti_disp[y_m][new_x] = old_disp[i][j] + flowL[i][j][0] - flowR[i][xr][0]
                        esti_disp[y_m][new_x] = max(0, esti_disp[y_m][new_x])
                    else:
                        esti_disp[y_m][new_x] = new_x - best_patch
                        esti_disp[y_m][new_x] = max(0, esti_disp[y_m][new_x])

            # print('[delta] time = %.2f' %(time.time() - t1))

        print('[update] time = %.2f' %(time.time() - start_time))
        # if want to check the motion
        if draw_motion:
            print("outL shape: ", outL.shape)
            skimage.io.imsave(str(index+ii).zfill(4)+"_left_flow.jpg", outL)
            # end of checking

        res = compareDisp(index+ii, esti_disp)
        compare_result.append(res)


print("[STEREO]",args.path,np.mean(compare_result))
