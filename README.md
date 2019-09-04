# ISM_algorithm

This ISM algorithm is a plug-in supportive repository to accelerate a general stereo vision DNN using Invariant-based Stereo Matching for continuous stereo vision application from our paper, *ASV: Accelerated Stereo Vision System*.

## How to use

In the `ism_skeleton.py`, it gives a skeleton code for using Invariant-based motion compensation with any Stereo DNN model. There are several steps in TODO list in order to make this algorithm works.

First, you need to download a stereo vision dataset. For this, we recommend [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php) and [SceneFlow dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and pass the path to the skeleton code.

Second, download any stereo vision DNN network to test this code. In the `ism_skeleton.py`, you need to implement the functions `load_dnn_model` and `dnn_inference` to be able to load the stereo DNN model and use the stereo DNN model to geneerate disparity results from key frames.

That's it!

## Scripts

In the directory `stereo_script`, it contains several scripts that we applied our methods on four representative DNNs. Here are some detailed instructions to run those scripts.

### PSMNet

[PSMNet](https://github.com/JiaRenChang/PSMNet) is from a 2018 CVPR paper by Jia-Ren Chang and Yong-Sheng Chen. 

To use our script, please first check out this [link](https://github.com/JiaRenChang/PSMNet) and follow the instructions to set up their model appropriately.

Next, you can download one of the stereo vision datasets. For the demostration purpose, we choose "[sceneflow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)". Download both "RGB images" and "Disparity" to the same directory `PSMNet` and untar them.

After untaring both "RGB image" and Disparity, you will see a `frames_cleanpass` folder and a `disparity` folder in the directory.

Next, copy two scripts from `psmnet` to their root directory, `PSMNet`. Then, you can run our algorithm by:

The `multi_frame_compensation_bm.py` is the modified script that add our ISM algorithm into their orignal test python script. The `test_mutli_bm.sh` is a bash script to test the ISM algorithm on different datasets in `frames_cleanpass` folder and place each test result into a folder named `result`.

A simple example of using `multi_frame_compensation_bm.py`:
```
  $ python multi_frame_compensation_bm.py \
       --maxdisp 192 --model stackhourglass \
       --loadmodel pretrained_model_KITTI2015.tar \
       --saveimg False --datapath $PATH \
       --datasize 100

```

You need to specify the name of dataset in the `frames_cleanpass` to run this python script. `datasize` flag specify the number of image you want to process in this dataset.

To test a set of dataset, you can run
```
  $ ./test_multi_bm.sh
```

You can also modified `test_multi_bm.sh` to specified the dataset that you want to test on our algorithm. After finished running the test script, you can check the result from `result` directory.

One of the example results is attached below: 

```
Number of model parameters: 5224768
(1, 3, 544, 960) (1, 3, 544, 960)
old_disp:  (540, 960)
time = 39.79
Total index: 2, EW: 1
(1, 3, 544, 960) (1, 3, 544, 960)
pred_disp:  (540, 960)
time = 39.12
old_comparison:  24.032894 2.670288e-05 3.2993202 ...
esti_comparison:  31.811045 5.2452087e-06 3.6652005 ...
pred_comparison:  23.935379 2.2888184e-05 3.3056526 ...

......
```

For every pair of frames, the result shows the execution time to generate its disparity map, the index of image pair, and accuracy comparing with ground truth. There are two key comparison results:
 * esti_comparison : the accuracy using our ISM algorithm.
 * pred_comparison : the accuracy using DNN.
 
The first three fields in each comparison stand for:
  1. the maximum difference between ground truth and result;
  2. the minimum difference between ground truth and result;
  3. the average difference between ground truth and result;


## Some details about this skeleton code

Inside of this skeleton script, we used `OpticalFlowFarneback` in OpenCV to compensate motions across adjacent frames。 Other dense optical flow algrithm can also be used to substitute this function in order to get the disparity map from next subsequent frames.

## Contact

If there is any question, please email me: yfeng28@ur.rochester.edu.
