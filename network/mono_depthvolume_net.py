# Copyright (c) 2023 42dot. All rights reserved.
import torch.nn as nn

from external.layers import ResnetEncoder, DepthDecoder
import torch
import torch.nn.functional as F

class SoftAttnDepth(nn.Module):
    def __init__(self, alpha, beta, discretization):
        super(SoftAttnDepth, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.discretization = discretization

    def get_depth_sid(self, depth_labels):
        alpha_ = torch.FloatTensor([self.alpha])
        beta_ = torch.FloatTensor([self.beta])
        t = []
        for K in range(depth_labels):
            K_ = torch.FloatTensor([K])
            t.append(torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * K_ / depth_labels))
        t = torch.FloatTensor(t)
        return t

    def forward(self, input_t, eps=1e-6):
        batch_size, depth, height, width = input_t.shape
        if self.discretization == 'SID':
            grid = self.get_depth_sid(depth).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            grid = torch.linspace(
                self.alpha, self.beta, depth,
                requires_grad=False).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        grid = grid.repeat(batch_size, 1, height, width).float()

        z = F.softmax(input_t, dim=1)
        z = z * (grid.to(z.device))
        z = torch.sum(z, dim=1, keepdim=True)

        return z

class MonoDepthVolumeNet(nn.Module):
    """
    Pytorch module for a depth network from the paper
    "Digging into Self-Supervised Monocular Depth Prediction"
    """

    def __init__(self, cfg):
        super(MonoDepthVolumeNet, self).__init__()
        self.num_layers = cfg['model']['num_layers']
        self.pretrained = cfg['model']['weights_init']
        self.scales = cfg['training']['scales']
        self.d_num = cfg['model']['d_num']
        self.d_type=cfg['model']['d_type']
        self.min_depth = cfg['training']['min_depth']
        self.max_depth = cfg['training']['max_depth']

        self.depth_encoder = ResnetEncoder(self.num_layers, self.pretrained, 1)
        del self.depth_encoder.encoder.fc  # For ddp training
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, self.scales,num_output_channels=self.d_num)
        self.depth_volume_layer =SoftAttnDepth(self.min_depth,self.max_depth,self.d_type)


    def forward(self, input_images):
        depth_feature = self.depth_encoder(input_images)
        outputs = self.depth_decoder(depth_feature)
        for scale in self.scales:
            outputs[("disp", scale)] = self.depth_volume_layer(outputs[("depth_volume", scale)])
        return outputs