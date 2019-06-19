#!/usr/bin/env bash

python src/pdet_test.py pdet \
--load_model exp/pdet/coco_dla_2x/model_best.pth  \
--dataset wider \
--gpus 0 \
--scores_thresh 0.5 \
--input_h 384 --input_w 768 \
--K 50
