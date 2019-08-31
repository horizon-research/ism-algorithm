# ISM_algorithm

This ISM algorithm is a plug-in supportive repository to accelerate a general stereo vision DNN using Invariant-based Stereo Matching for continuous stereo vision application from our paper, *ASV: Accelerated Stereo Vision System*.

## How to use

In the `ism_skeleton.py`, it gives a skeleton code for using Invariant-based motion compensation with any Stereo DNN model. There are several steps in TODO list in order to make this algorithm works.

First, you need to download a stereo vision dataset. For this, we recommend [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php) and [SceneFlow dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and pass the path to the skeleton code.

Second, download any stereo vision DNN network to test this code. In the `ism_skeleton.py`, you need to implement the functions `load_dnn_model` and `dnn_inference` to be able to load the stereo DNN model and use the stereo DNN model to geneerate disparity results from key frames.

That's it!

## Some details about this skeleton code

Inside of this skeleton script, we used `OpticalFlowFarneback` in OpenCV to compensate motions across adjacent frames。 Other dense optical flow algrithm can also be used to substitute this function in order to get the disparity map from next subsequent frames.

## Contact

If there is anything questions, please email me: yfeng28@ur.rochester.edu.
