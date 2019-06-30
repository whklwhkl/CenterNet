#!/usr/bin/env bash

python src/main.py pdet_norm \
--dataset wider \
--exp_id wider2019pd_raw_608_1216 \
--batch_size 64 \
--input_h 256 --input_w 512 \
--lr 1e-3 \
--gpus 0 \
--num_workers 8 \
--num_epochs 10 \
--lr_step 80 \
--val_intervals 1 \
--metric ap \
--K 50