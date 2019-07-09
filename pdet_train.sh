#!/usr/bin/env bash

python src/main.py pdet \
--dataset wider \
--exp_id wider2019pd_raw_608_1216 \
--batch_size 16 \
--input_h 608 --input_w 1216 \
--lr 5e-5 \
--gpus 0 \
--num_workers 2 \
--num_epochs 10 \
--lr_step 80 \
--val_intervals 1 \
--metric ap \
--K 50 \
--load_model /home/wanghao/PycharmProjects/reid/CenterNet/exp/pdet/wider2019pd_raw_480_960/model_ap549.pth