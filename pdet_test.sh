#!/usr/bin/env bash

python src/pdet_test.py pdet \
--load_model exp/pdet/wider2019pd_raw_384_768/model_last.pth  \
--dataset wider \
--gpus 0 \
--scores_thresh 0.5 \
--input_h 384 --input_w 768 \
--K 50

zip -j sub.zip /home/wanghao/PycharmProjects/reid/CenterNet/exp/pdet/default/submission.txt
echo $(pwd)/sub.zip