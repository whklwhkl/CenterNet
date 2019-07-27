#!/usr/bin/env bash

python src/pdet_test.py pdet \
--load_model exp/pdet/wider2019pd_raw_608_1216/model_best.pth  \
--dataset wider \
--gpus 0 \
--scores_thresh 0.5 \
--input_h 1216 --input_w 1216 \
--K 50
#--input_h 608 --input_w 1216 \
#--test_scales 0.25,0.5,0.75,1.0
#--flip_test
#--load_model /home/wanghao/PycharmProjects/reid/CenterNet/exp/pdet/wider2019pd_raw_480_960/model_ap549.pth  \

#zip -j sub.zip /home/wanghao/PycharmProjects/reid/CenterNet/exp/pdet/default/submission.txt
#echo $(pwd)/sub.zip
#--gpus 0