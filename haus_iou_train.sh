#!/usr/bin/env bash

python src/main.py haus \
--dataset wider \
--exp_id wider2019pd_raw_448_896 \
--batch_size 24 \
--input_h 448 --input_w 896 \
--lr 7e-5 \
--gpus 0 \
--num_workers 2 \
--num_epochs 10 \
--lr_step 80 \
--val_intervals 1 \
--metric ap \
--K 50 \
--hm_weight 0.01 \
--wh_weight 1.0 \
--load_model /home/wanghao/PycharmProjects/reid/CenterNet/exp/pdet/wider2019pd_raw_416_800/model_ap537.pth