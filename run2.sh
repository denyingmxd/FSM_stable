#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

#python -W ignore train.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_align_vertical_mask.yaml'
#
#python -W ignore eval.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_align_vertical_mask.yaml'
#
#python -W ignore train.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_no_IA_no_temporal_align_vertical_mask.yaml'

python -W ignore eval.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_no_IA_no_temporal_align_vertical_mask.yaml'
#




