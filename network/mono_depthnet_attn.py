# Copyright (c) 2023 42dot. All rights reserved.
import torch.nn as nn

from external.layers import ResnetEncoder, DepthDecoder


class MonoDepthNet_attn(nn.Module):
    """
    Pytorch module for a depth network from the paper
    "Digging into Self-Supervised Monocular Depth Prediction"
    """

    def __init__(self, cfg):
        super(MonoDepthNet_attn, self).__init__()
        num_layers = cfg['model']['num_layers']
        pretrained = cfg['model']['weights_init']
        scales = cfg['training']['scales']
        self.depth_encoder = ResnetEncoder(num_layers, pretrained, 1)
        del self.depth_encoder.encoder.fc  # For ddp training
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, scales)
        self.multihead_attn = nn.MultiheadAttention(self.depth_encoder.num_ch_enc[-1], 8)

    def forward(self, input_images):
        depth_feature = self.depth_encoder(input_images)
        last_feature = depth_feature[-1]
        b,c,h,w = last_feature.shape#SNE
        last_feature_patches = last_feature.reshape(b, c, h * w).permute((2, 0, 1))
        last_feature_patches,last_feature_patches_weight = \
            self.multihead_attn(last_feature_patches,last_feature_patches,last_feature_patches)
        depth_feature[-1] = last_feature_patches.permute((1,2,0)).reshape(b,c,h,w)

        outputs = self.depth_decoder(depth_feature)

        return outputs