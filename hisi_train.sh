#!/usr/bin/env bash

set -e

function stage(){
let w=$1*2
python src/main.py pdet \
--arch res_50 \
--dataset wider \
--exp_id wider2019pd_raw_$1 \
--batch_size $2 \
--input_h $1 --input_w $w \
--lr $3 \
--gpus 0 \
--num_workers 2 \
--num_epochs $4 \
--lr_step 80 \
--val_intervals 1 \
--metric ap \
--K 50 \
--load_model $5
}

#   height  bs  lr      ep  load_model_path
#stage 256   64  2e-4    40  _
#stage 320   32  1e-4    20  exp/pdet/wider2019pd_raw_256/model_best.pth
#stage 384   32  1e-4    20  exp/pdet/wider2019pd_raw_320/model_best.pth
#stage 416   16  5e-5    20  exp/pdet/wider2019pd_raw_384/model_best.pth
#stage 448   16  5e-5    20  exp/pdet/wider2019pd_raw_416/model_best.pth
#stage 480   16  4e-5    20  exp/pdet/wider2019pd_raw_448/model_best.pth
#stage 512   8   2e-5    20  exp/pdet/wider2019pd_raw_480/model_best.pth
#stage 544   8   2e-5    20  exp/pdet/wider2019pd_raw_512/model_best.pth
#stage 576   8   1e-5    20  exp/pdet/wider2019pd_raw_544/model_best.pth
#stage 608   8   1e-5    20  exp/pdet/wider2019pd_raw_576/model_best.pth
#stage 640   8   5e-6    20  exp/pdet/wider2019pd_raw_608/model_best.pth
stage 672   8   3e-6    20  exp/pdet/wider2019pd_raw_640/model_best.pth
