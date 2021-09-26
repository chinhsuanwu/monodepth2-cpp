# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseDecoder(nn.Module):
    def __init__(self, num_input_features=1, num_frames_to_predict_for=2, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.pconv0 = nn.Conv2d(512, 256, 1)
        self.pconv1 = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.pconv2 = nn.Conv2d(256, 256, 3, stride, 1)
        self.pconv3 = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [F.relu(self.pconv0(f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        out = F.relu(self.pconv1(out))
        out = F.relu(self.pconv2(out))
        out = self.pconv3(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

if __name__ == '__main__':
    device = torch.device('cuda')
    pose_decoder = PoseDecoder().to(device)
    
    inputs = []
    inputs.append(torch.zeros(8, 64, 96, 320).to(device))
    inputs.append(torch.zeros(8, 64, 48, 160).to(device))
    inputs.append(torch.zeros(8, 128, 24, 80).to(device))
    inputs.append(torch.zeros(8, 256, 12, 40).to(device))
    inputs.append(torch.zeros(8, 512, 6, 20).to(device))
    inputs = [inputs]
    
    output = pose_decoder(inputs)
    pose_decoder = torch.jit.script(pose_decoder)
    torch.jit.save(pose_decoder, 'pose_decoder.jit.pt')