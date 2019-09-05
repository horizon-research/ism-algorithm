#!/bin/bash

# different directory a.k.a. different dataset
declare -a arr=( "lonetree_augmented0_x2" \
                 "eating_x2" "lonetree_x2" \
                 "treeflight_x2" \
                 "a_rain_of_stones_x2" \
                 "family_x2" "lonetree_difftex2_x2" \
                 "lonetree_difftex_x2" \
                 "treeflight_augmented0_x2")

PATH=.

## now loop through the above array
for i in "${arr[@]}"
do
   echo ${i}
   ./gcnet_with_ism.py \
       --ew 4 --path ${PATH}/${i}/ > result/${i}.txt
done

