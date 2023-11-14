#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -W ignore train.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_ailgn_depth_con_nearest_999_two_view_loss_0.001.yaml'

python -W ignore eval.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_ailgn_depth_con_nearest_999_two_view_loss_0.001.yaml'
