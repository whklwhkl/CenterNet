#!/usr/bin/env bash

python src/main.py pdet \
--dataset wider \
--exp_id wider2019pd_raw_384_768 \
--batch_size 32 \
--input_h 384 --input_w 768 \
--lr 1e-2 \
--gpus 0 \
--num_workers 8 \
--num_epochs 32 \
--lr_step 8,16 \
--val_intervals 1