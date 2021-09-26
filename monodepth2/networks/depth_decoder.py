# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from typing import Dict, List

from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.upconv5 = ConvBlock(512, 256)
        self.upconv4 = ConvBlock(256, 128)
        self.upconv3 = ConvBlock(128, 64)
        self.upconv2 = ConvBlock(64, 32)
        self.upconv1 = ConvBlock(32, 16)

        self.iconv5 = ConvBlock(512, 256)
        self.iconv4 = ConvBlock(256, 128)
        self.iconv3 = ConvBlock(128, 64)
        self.iconv2 = ConvBlock(96, 32)
        self.iconv1 = ConvBlock(16, 16)

        self.disp4 = Conv3x3(128, self.num_output_channels)
        self.disp3 = Conv3x3(64, self.num_output_channels)
        self.disp2 = Conv3x3(32, self.num_output_channels)
        self.disp1 = Conv3x3(16, self.num_output_channels)

    def forward(self, input_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        outputs = {}

        x = input_features[4]
        x = self.upconv5(x)
        x = [self.upsample(x)]
        if self.use_skips:
            x += [input_features[3]]
        x = torch.cat(x, 1)
        x = self.iconv5(x)

        x = self.upconv4(x)
        x = [self.upsample(x)]
        if self.use_skips:
            x += [input_features[2]]
        x = torch.cat(x, 1)
        x = self.iconv4(x)

        outputs['disp3'] = torch.sigmoid(self.disp4(x))

        x = self.upconv3(x)
        x = [self.upsample(x)]
        if self.use_skips:
            x += [input_features[1]]
        x = torch.cat(x, 1)
        x = self.iconv3(x)

        outputs['disp2'] = torch.sigmoid(self.disp3(x))

        x = self.upconv2(x)
        x = [self.upsample(x)]
        if self.use_skips:
            x += [input_features[0]]
        x = torch.cat(x, 1)
        x = self.iconv2(x)

        outputs['disp1'] = torch.sigmoid(self.disp2(x))

        x = self.upconv1(x)
        x = [self.upsample(x)]
        x = torch.cat(x, 1)
        x = self.iconv1(x)

        outputs['disp0'] = torch.sigmoid(self.disp1(x))

        return outputs


if __name__ == '__main__':
    device = torch.device('cuda')
    depth_decoder = DepthDecoder().to(device)
    
    inputs = []
    inputs.append(torch.zeros(8, 64, 96, 320).to(device))
    inputs.append(torch.zeros(8, 64, 48, 160).to(device))
    inputs.append(torch.zeros(8, 128, 24, 80).to(device))
    inputs.append(torch.zeros(8, 256, 12, 40).to(device))
    inputs.append(torch.zeros(8, 512, 6, 20).to(device))
    
    output = depth_decoder(inputs)
    depth_decoder = torch.jit.script(depth_decoder)
    torch.jit.save(depth_decoder, 'depth_decoder.jit.pt')