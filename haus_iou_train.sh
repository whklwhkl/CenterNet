#!/usr/bin/env bash

python src/main.py haus \
--dataset wider \
--exp_id wider2019pd_raw_480_960 \
--batch_size 24 \
--input_h 480 --input_w 960 \
--lr 7e-5 \
--gpus 0 \
--num_workers 2 \
--num_epochs 10 \
--lr_step 80 \
--val_intervals 1 \
--metric ap \
--K 50 \
--hm_weight 1.0 \
--wh_weight 1.0 \
--load_model /home/wanghao/PycharmProjects/reid/CenterNet/exp/pdet/wider2019pd_raw_480_960/model_ap549.pth
#--num_iters 32 \
