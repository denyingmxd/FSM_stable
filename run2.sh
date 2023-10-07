#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

#python -W ignore train.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_depth_con.yaml'

python -W ignore eval.py \
--config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_ailgn_depth_con.yaml' \
--weight_path='/data/laiyan/codes/FSM_stable/results/baseline_384_front_border_scale_IA_no_temporal_ailgn_depth_con/models/weights_19'

python -W ignore eval.py \
--config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_ailgn_depth_con.yaml' \
--weight_path='/data/laiyan/codes/FSM_stable/results/baseline_384_front_border_scale_IA_no_temporal_ailgn_depth_con/models/weights_19' \
--post_process

