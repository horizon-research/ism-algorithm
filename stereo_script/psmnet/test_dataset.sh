#!/bin/bash


# different directory a.k.a. different dataset
declare -a arr=("eating_naked_camera2_x2" \
                "flower_storm_augmented1_x2" \
                "funnyworld_camera2_augmented0_x2" \
                "lonetree_augmented0_x2"  "lonetree_winter_x2"\
                "treeflight_augmented1_x2" "eating_x2"  \
                "flower_storm_x2"  "lonetree_x2" "treeflight_x2" \
                "a_rain_of_stones_x2"  "family_x2" \
                "funnyworld_augmented0_x2" "funnyworld_camera2_x2" \
                "lonetree_difftex2_x2" "top_view_x2"
                "eating_camera2_x2" "flower_storm_augmented0_x2" \
                "funnyworld_x2" "lonetree_difftex_x2" \
                "treeflight_augmented0_x2")

# source /home/tigris/parsec-3.0/env.sh

## now loop through the above array
for i in "${arr[@]}"
do
   python multi_frame_compensation_bm.py \
       --maxdisp 192 --model stackhourglass \
       --loadmodel pretrained_model_KITTI2015.tar \
       --saveimg False --datapath ${i}/ \
       --datasize 100 > result/${i}.txt

done

