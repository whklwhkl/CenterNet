#!/usr/bin/env bash

python src/main.py pdet \
--dataset wider \
--exp_id coco_dla_2x \
--batch_size 32 \
--lr 5e-4 \
--gpus 0 \
--num_workers 8 \
--num_epochs 230 \
--lr_step 180,210