from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import skimage.transform
from numpy import *
import numpy as np
import time
import math
import re
import sys
from utils import preprocess
from models import *
import cv2 as cv

# parse the arguments
parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default=None,
                    help='select model with no "/" at front')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--saveimg', type=bool, default=False,
					help='save processed images')
parser.add_argument('--datasize', type=int, default=100,
					help='data size of the imgs')
parser.add_argument('--ew', type=int, default=4,
                    help='extrapolate window')
parser.add_argument('--p_size', type=int, default=7,
                    help='extrapolate window')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cpu()

datapath = None
if args.datapath is None:
	print('None data path')
	exit(-1)
else:
	datapath = args.datapath

save_img = args.saveimg
data_size = args.datasize

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel, map_location='cpu')
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

'''
Original function in previous paper to do prediction,
and then, return the predicted disparity map from PMSnet.
'''
def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = torch.FloatTensor(imgL).cuda()
           imgR = torch.FloatTensor(imgR).cuda()
        else:
           imgL = torch.FloatTensor(imgL).cpu()
           imgR = torch.FloatTensor(imgR).cpu()

        imgL, imgR= Variable(imgL), Variable(imgR)

        print(imgL.shape, imgR.shape)

        with torch.no_grad():
            output = model(imgL,imgR)
        output = torch.squeeze(output)
        pred_disp = output.data.cpu().numpy()

        return pred_disp

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

'''
Main function to launch the program
'''
def main():

    # initialization
    processed = preprocess.get_transform(augment=False)

    # some constants and settings
    draw_motion = False
    prediction = True
    do_denoise = False
    # some configs
    bar = 5
    ew = args.ew
    ew = 1
    saved_dir = "result/"
    left_prefix = "frames_cleanpass/"+args.datapath+"left/"
    right_prefix = "frames_cleanpass/"+args.datapath+"right/"
    result_prefix = "disparity/"+args.datapath+"left/"

	# data collector
    collected_result = {"old_err": [],
                        "esti_err": [],
                        "pred_err": [],
                        "old_rate": [],
                        "esti_rate": [],
                        "pred_rate": []}


    # initial old_disp to None
    old_disp = None

    # start the core routine
    for inx in range(1, data_size-ew, ew):
		# avoid recomputation if we have already compute this.
        if old_disp is None:
	        # print(res_disp)
    	    oldL_o = (skimage.io.imread(left_prefix + str(inx).zfill(4) + ".png").astype('float32'))
            oldR_o = (skimage.io.imread(right_prefix + str(inx).zfill(4) + ".png").astype('float32'))
            oldL = processed(oldL_o).numpy()
            oldR = processed(oldR_o).numpy()
            oldL = np.reshape(oldL,[1,3,oldL.shape[1],oldL.shape[2]])
            oldR = np.reshape(oldR,[1,3,oldR.shape[1],oldR.shape[2]])

            # padding zeros to the image edges
            top_pad = 544 - oldL.shape[2]
            left_pad = 960 - oldL.shape[3]

            oldL = np.lib.pad(oldL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
            oldR = np.lib.pad(oldR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

            # test the first pair of flames
            start_time = time.time()
            old_disp = test(oldL, oldR)
            # reshape back
            old_disp = old_disp[top_pad:,:]
            print('old_disp: ', old_disp.shape)
            print('time = %.2f' %(time.time() - start_time))

		## End of init old_disp ##

        # resize the data for fast estimation if wanted
        resize_delta = 1.0

        # old grey image
        oldL_grey = cv.imread(left_prefix + str(inx).zfill(4) + ".png", 0)
        oldR_grey = cv.imread(right_prefix + str(inx).zfill(4) + ".png", 0)
        # old grey image resize
        oldL_grey = cv.resize(oldL_grey, (0,0), fx = resize_delta, fy = resize_delta)
        oldR_grey = cv.resize(oldR_grey, (0,0), fx = resize_delta, fy = resize_delta)

        for ii in range(1, ew+1):
            print("Total index: %d, EW: %d" % (inx+ii, ii))
		    # start for the new prediction routine
            imgL_o = (skimage.io.imread(left_prefix + str(inx+ii).zfill(4) + ".png").astype('float32'))
            imgR_o = (skimage.io.imread(right_prefix + str(inx+ii).zfill(4) + ".png").astype('float32'))
            imgL = processed(imgL_o).numpy()
            imgR = processed(imgR_o).numpy()
            imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
            imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])

            imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
            imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)

            # test the second pair of frames;
            start_time = time.time()
            pred_disp = test(imgL,imgR)
            # reshape back
            pred_disp = pred_disp[top_pad:,:]
            print('pred_disp: ', pred_disp.shape)
            print('time = %.2f' %(time.time() - start_time))

            # curr image
            imgL_grey = cv.imread(left_prefix + str(inx+ii).zfill(4) + ".png", 0)
            imgR_grey = cv.imread(right_prefix + str(inx+ii).zfill(4) + ".png", 0)
            # curr image resize
            imgL_grey = cv.resize(imgL_grey, (0,0), fx = resize_delta, fy = resize_delta)
            imgR_grey = cv.resize(imgR_grey, (0,0), fx = resize_delta, fy = resize_delta)

            # openCV return matrix of (x, y)
            flowL = cv.calcOpticalFlowFarneback(oldL_grey,imgL_grey, None, 0.5, 4, 16, 5, 5, 1.2, 0)
            flowR = cv.calcOpticalFlowFarneback(oldR_grey,imgR_grey, None, 0.5, 4, 16, 5, 5, 1.2, 0)

            # copy the disp and proceed predict part
            esti_disp = old_disp.copy()

            # this two are used to draw motions
            outL = cv.imread(left_prefix + str(inx).zfill(4) + ".png", 0)
            outL = cv.resize(outL, (0,0), fx = resize_delta, fy = resize_delta).astype(float)
            outR = cv.imread(right_prefix + str(inx).zfill(4) + ".png", 0)
            outR = cv.resize(outR, (0,0), fx = resize_delta, fy = resize_delta).astype(float)

            # this is just check the accuracy of the optical flow;
            select_idx = int(resize_delta*16)
            for i in range(len(flowL)):
                for j in range(len(flowL[0])):
                    if draw_motion and i % select_idx == 0 and j % select_idx  == 0:
                        # draw line to the output;
                        xl = int(min(len(flowL[0])-1, max(0, j+flowL[i][j][0])))
                        yl = int(min(len(flowL)-1, max(0, i+flowL[i][j][1])))
                        xr = int(min(len(flowR[0])-1, max(0, j+flowR[i][j][0])))
                        yr = int(min(len(flowR)-1, max(0, i+flowR[i][j][1])))
                        # outL[i][j] = sqrt(flowL[i][j][0]*flowL[i][j][0]+flowL[i][j][1]*flowL[i][j][1])

                        # draw lines in the images;
                        cv.line(outL, (j, i), (xl, yl), (0,255,0), 1)
                        cv.line(outR, (j, i), (xr, yr), (0,255,0), 1)

                    # update the flow new indexes;
                    flowL[i][j][0] = int(flowL[i][j][0])
                    flowL[i][j][1] = int(flowL[i][j][1])
                    flowR[i][j][0] = int(flowR[i][j][0])
                    flowR[i][j][1] = int(flowR[i][j][1])


                    #########################################
                    #        set  the search config         #
                    BM_search = True
                    p_size = args.p_size
                    #########################################

                    # check if predict
                    if prediction:
                        xr = int(j-old_disp[i][j])
                        yl = int(min(len(flowL)-1, max(0, i+flowL[i][j][1])))
                        yr = int(min(len(flowR)-1, max(0, i+flowR[i][j][1])))
                        y_m = int((yl+yr)/2)
                        new_x = int(j+flowL[i][j][0])
                        if new_x >= 0 and new_x < len(flowL[0]) and xr >= 0 and xr < len(flowL[0]):
                            ########################### add BM search  ############################
                            if BM_search:
                                xl_patch = new_x
                                xr_patch = int(xr + flowL[i][j][0])
                                best_index = -(2*p_size+1)
                                best_patch = xr_patch
                                best_result = float('Inf')

                                for i_p in range(-3*p_size, 3*p_size):
                                    xl_p = xl_patch
                                    xr_p = xr_patch + i_p
                                    if xr_p >= 0 and xr_p + p_size < len(flowL[0]) and \
                                            xl_p >= 0 and xl_p + p_size < len(flowL[0]) \
                                            and y_m >= 0 and y_m + 4 < len(flowL):
                                        # extract the patches;
                                        # print((y_m, y_m+4), (xr_p, xr_p+p_size), (xl_p, xl_p+p_size))
                                        r_patch = outR[y_m : y_m+4, xr_p : xr_p + p_size]
                                        l_patch = outL[y_m : y_m+4, xl_p : xl_p + p_size]
                                        # print(xl_p, xr_p, l_patch, r_patch)

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

                            #######################################################################

                            # do regular motion compensate
                            else:
                                # update values;
                                esti_disp[y_m][new_x] = old_disp[i][j] + flowL[i][j][0] - flowR[i][xr][0]
                                esti_disp[y_m][new_x] = max(0, esti_disp[y_m][new_x])

            # if want to check the motion
            if draw_motion:
                print("outL shape: ", outL.shape)
                skimage.io.imsave(saved_dir + str(inx)+"_left_flow.jpg", outL)
                skimage.io.imsave(saved_dir + str(inx)+"_right_flow.jpg", outR)
            # end of checking

            # do denoising;
            if do_denoise:
                skimage.io.imsave("tmp.jpg", (esti_disp).astype('uint8'))
                tmp_img = cv.imread("tmp.jpg", 0)
                esti_disp = cv.fastNlMeansDenoising(tmp_img, None, 3, 9, 21)

            # open the disparity result
            file = open(result_prefix + str(inx+ii).zfill(4) + ".pfm", "r")
            res_disp, _ = load_pfm(file)
            res_disp = np.flip(res_disp, 0)

            # analysis
            mean_v, err_v = compare_imgs(old_disp, res_disp, bar, "old")
            collected_result["old_err"].append(mean_v)
            collected_result["old_rate"].append(err_v)
            mean_v, err_v = compare_imgs(esti_disp, res_disp, bar, "esti")
            collected_result["esti_err"].append(mean_v)
            collected_result["esti_rate"].append(err_v)
            mean_v, err_v = compare_imgs(pred_disp, res_disp, bar, "pred")
            collected_result["pred_err"].append(mean_v)
            collected_result["pred_rate"].append(err_v)

            # save the resulted images
            if save_img:
                skimage.io.imsave(saved_dir+str(inx)+"_old.jpg", (old_disp).astype('uint8'))
                skimage.io.imsave(saved_dir+str(inx)+"_esti.jpg", (esti_disp).astype('uint8'))
                skimage.io.imsave(saved_dir+str(inx)+"_pred.jpg", (pred_disp).astype('uint8'))

		    # copy the previous pred_disp for next prediction
            old_disp = pred_disp.copy()


    for k in collected_result:
        print("[STEREO]",k, np.mean(collected_result[k]))

if __name__ == '__main__':
   main()



# python motion_compensation.py --maxdisp 192 --model stackhourglass --datapath  --loadmodel pretrained_sceneflow.tar --saveimg True


