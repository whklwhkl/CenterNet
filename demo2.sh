#!/usr/bin/env bash

python src/demo.py pdet \
--demo /home/wanghao/PycharmProjects/reid/CenterNet/images/17790319373_bd19b24cfc_k.jpg \
--load_model /home/wanghao/PycharmProjects/reid/CenterNet/exp/pdet/wider2019pd_raw_608_1216/dla34_ap567.pth \
--scores_thresh 0.5 \
--input_h 1216 --input_w 1216