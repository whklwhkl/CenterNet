#!/usr/bin/env bash

python src/main.py pdet_logwh \
--dataset wider \
--exp_id diff \
--batch_size 32 \
--input_h 352 --input_w 704 \
--lr 5e-4 \
--gpus 0 \
--num_workers 2 \
--num_epochs 10 \
--lr_step 80 \
--val_intervals 1 \
--metric ap \
--wh_weight 1.0 \
--K 50 \
--load_model /home/wanghao/PycharmProjects/reid/CenterNet/exp/pdet_logwh/diff/model_best.pth
#--hm_weight 1.0 \
#--off_weight 1.0