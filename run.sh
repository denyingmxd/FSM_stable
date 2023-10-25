#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

#python -W ignore train.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_align_fusion_128.yaml'
#python -W ignore eval.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_align_fusion_128.yaml'

#python -W ignore train.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_ailgn_depth_con_flipv5_sptp_con_0.2.yaml'
#python -W ignore eval.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_ailgn_depth_con_flipv5_sptp_con_0.2.yaml'


python -W ignore eval.py --config_file='./configs/ddp/ddad/baseline_384_front_border_scale_IA_no_temporal_align_depth_con_depth_smooth_0.01.yaml'
