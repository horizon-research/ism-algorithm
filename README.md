# ISM_algorithm

This ISM algorithm is an algorithm framework to accelerate a general stereo vision DNN using Invariant-based Stereo Matching for continuous stereo vision application from our paper, *ASV: Accelerated Stereo Vision System*.

We provide some implementation of ISM algorithm for several popular stereo DNNs.

  * [PSMNet](#psmnet): ["Pyramid Stereo Matching Network"](https://arxiv.org/abs/1803.08669)
  * [FlowNet/DispNet](#flownetdispnet): ["FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks"](https://arxiv.org/abs/1612.01925)
  * [GC-Net](#gc-net): ["End-to-End Learning of Geometry and Context for Deep Stereo Regression"](https://arxiv.org/abs/1703.04309)

## A generic framework for stereo DNN

In the `ism_skeleton.py`, it gives a skeleton code for using Invariant-based motion compensation with any Stereo DNN model. It contains the main bone to implement our ISM algorithm to other stereo DNNs. There are several steps in TODO list in order to make this algorithm works.

First, you need to download a stereo vision dataset. For this, we recommend [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_depth_all.php) and [SceneFlow dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and pass the path to the skeleton code.

Second, download any stereo vision DNN network to test this code. In the `ism_skeleton.py`, you need to implement the functions `load_dnn_model` and `dnn_inference` to be able to load the stereo DNN model and use the stereo DNN model to geneerate disparity results from key frames.

### Some details about this skeleton code

Inside of this skeleton script, we used `OpticalFlowFarneback` in OpenCV to compensate motions across adjacent frames. Other dense optical flow algrithm can also be used to substitute this function. The optical flow algorithm is used in one stage of ISM algorithm to get the disparity map from next subsequent frames.

## Implementation for popular stereo DNNs

In the directory `stereo_script`, it contains several scripts that we applied our methods on four representative stereo DNNs. Here are some detailed instructions to run these scripts.

### PSMNet

[PSMNet](https://github.com/JiaRenChang/PSMNet) is from a 2018 CVPR paper by Jia-Ren Chang and Yong-Sheng Chen. 

To use our script, please first check out this [link](https://github.com/JiaRenChang/PSMNet) and follow the instructions to set up PSMNet model appropriately.

```
  $ git clone https://github.com/JiaRenChang/PSMNet.git
  $ cd PSMNet
```

Next, you can download one of the stereo vision datasets. For the demostration purpose, we choose "[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)". Download both "RGB images" and "Disparity" into the same directory `PSMNet` and untar them.

```
  $ tar -xzvf monkaa_frames_cleanpass.tar
  $ tar -xzvf monkaa_disparity.tar.bz2
```

After untaring both "RGB image" and "Disparity" tar, you will see two folders, `frames_cleanpass` and `disparity`, in the directory.

Next, copy two scripts from `script/psmnet` from our repository to their root directory, `PSMNet`. `psmnet_with_ism.py` is a modified script that adds our ISM algorithm into PSMNet orignal python script. The `test_dataset.sh` is a bash script to test the ISM algorithm on different datasets in `frames_cleanpass` folder and place each test result into a folder named `result`.

A simple example of using `psmnet_with_ism.py`:
```
  $ python psmnet_with_ism.py \
       --maxdisp 192 --model stackhourglass \
       --loadmodel pretrained_model_KITTI2015.tar \
       --saveimg False --datapath ${PATH} \
       --datasize 100
```

You need to specify the name of dataset as `${PATH}` in the `frames_cleanpass` to run this python script. `datasize` flag specifies the number of images that you want to process in this dataset. 

To test a set of dataset, you can run
```
  $ ./test_dataset.sh
```

You can also modified `test_dataset.sh` to specify the dataset that you want to test on. After finished running the test script, you can check all test results from `result` directory.

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

### FlowNet/DispNet

To run FlowNet and DispNet along with our ISM algorithm, please first clone and setup [Flownet2](https://github.com/lmb-freiburg/flownet2) from the paper by E. Ilg et al. 

```
  $ git clone https://github.com/lmb-freiburg/flownet2.git
  $ cd flownet2
```

Assuming now you are in the FlowNet root directory, after you get FlowNet running successfully, copy the two scripts from `script/flownet` to their `./scripts` directory.

And then, download the "[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)" dataset to the root directory, `flownet2`,  and untar the dataset.

```
  $ tar -xzvf monkaa_frames_cleanpass.tar
  $ tar -xzvf monkaa_disparity.tar.bz2
```

To run the ISM algorithm with FlowNet-C, run the command below:

```
  $ ./flownet_with_ism.py \
         ../models/FlowNet2-c/FlowNet2-c_weights.caffemodel \
         ../models/FlowNet2-c/FlowNet2-c_deploy.prototxt.template \
         --ew 4 --path ${PATH}

```
You can switch to DispNet by replace `FlowNet2-c` to `FLowNet2-S`.

You need to specify the path to the dataset, in this case, it is the root directory of flownet2.

There is also a batch-processing Bash script you can use to test on a set of datasets. Just simply run:
```
  $ ./test_dataset.sh
```
Then, you can check out results from `result` directory as previously shown. The format of result is the same as the one in PSMNet.


### GC-Net

GC-Net is from a 2017 CVPR paper: *End-to-End Learning of Geometry and Context for Deep Stereo Regression* by A. Kendall et al. We used the TensorRT version from NVDIA. To check out the stereo DNN, please to go this [link](https://github.com/NVIDIA-AI-IOT/redtail). Clone this repo:
```
  $ git clone https://github.com/NVIDIA-AI-IOT/redtail.git 
```
And then, redirect to the `redtail/stereoDNN` directory.
```
  $ cd redtail/stereoDNN
```
And follow the instructions to build the inference code (release version). 
After build succeeds, you should see a `nvstereo` binary in `bin` directory.

Now, you can make a new directory in the root directory to test our ISM algorithm.
```
  $ mkdir ism
```

Then, copy both binary and its weights into this directory.
```
  $ cp bin/nvstereo ism/nvstereo
  $ cp models/ResNet-18/TensorRT/trl_weights.bin ism/trl_weights.bin
```

Next, you can follow the previous instruction to download the dataset. 
For the demostration purpose, we choose "[SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)". Download 
both "RGB images" and "Disparity" in the same directory `PSMNet` and untar them.

```
  $ tar -xzvf monkaa_frames_cleanpass.tar
  $ tar -xzvf monkaa_disparity.tar.bz2
```

After untaring both "RGB image" and "Disparity" tar, you will see two folders,
`frames_cleanpass` and `disparity`, in the directory.

Last, copy two script from our directory, `stereo_script/gcnet`, to `ism` directory.
Now, you can run our ism algorithm using:
```
$ ./gcnet_with_ism.py \
       --ew 4 --path ${PATH}
```
The ${PATH} is the path lead to one particular dataset.

You can also use our script `test_dataset` to run multiple dataset together.

## Citing

This project is a artifact of our 2019 MICRO paper:

Y. Feng,  P. Whatmough, and Y. Zhu, "ASV: Accelerated Stereo Vision System", In Proc. of MICRO, 2019.

Please kindly consider citing this paper in your publications if it helps your research.
```
@inproceedings{yu2019asv,
  title={ASV: Accelerated Stereo Vision System},
  author={Feng, Yu and Whatmough, Paul and Zhu, Yuhao},
  booktitle={Proceedings of the 52th International Symposium on Microarchitecture},
  year={2019},
  organization={ACM}
}
```

## Contact

If there is any question, please email me: yfeng28@ur.rochester.edu.
