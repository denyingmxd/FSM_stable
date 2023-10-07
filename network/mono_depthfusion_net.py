# Copyright (c) 2023 42dot. All rights reserved.
import torch.nn as nn

from external.layers import ResnetEncoder
import torch
import torch.nn.functional as F
import numpy as np
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


# Copyright 2020 Toyota Research Institute.  All rights reserved.

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py


from collections import OrderedDict
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")




class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv_volume", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.convs[("dispconv_scalar", s)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:

                self.outputs[("depth_volume", i)] = self.convs[("dispconv_volume", i)](x)

                self.outputs[("disp_scalar", i)] = self.sigmoid(self.convs[("dispconv_scalar", i)](x))

        return self.outputs






class MonoDepthFusionNet(nn.Module):
    """
    Pytorch module for a depth network from the paper
    "Digging into Self-Supervised Monocular Depth Prediction"
    """

    def __init__(self, cfg):
        super(MonoDepthFusionNet, self).__init__()
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
        self.depth_volume_layer = SoftAttnDepth(self.min_depth,self.max_depth,self.d_type)

    def disp_to_depth(self,disp_in):
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        disp_range = max_disp - min_disp

        # disp_in = F.interpolate(disp_in, [self.height, self.width], mode='bilinear', align_corners=False)
        disp = min_disp + disp_range * disp_in
        depth = 1 / disp
        return depth

    def forward(self, input_images):
        depth_feature = self.depth_encoder(input_images)
        outputs = self.depth_decoder(depth_feature)
        for scale in self.scales:
            aa = self.depth_volume_layer(outputs[("depth_volume", scale)])
            bb = self.disp_to_depth(outputs[("disp_scalar", scale)])
            outputs[("disp", scale)] = 0.5*(aa+bb)
        return outputs