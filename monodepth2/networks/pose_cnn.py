# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames=2):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        self.conv1 = nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3)
        self.conv2 = nn.Conv2d(16, 32, 5, 2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 2, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 2, 1)

        self.pose_conv = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

    def forward(self, out):
        out = F.relu(self.conv1(out), inplace=True)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.relu(self.conv3(out), inplace=True)
        out = F.relu(self.conv4(out), inplace=True)
        out = F.relu(self.conv5(out), inplace=True)
        out = F.relu(self.conv6(out), inplace=True)
        out = F.relu(self.conv7(out), inplace=True)

        out = self.pose_conv(out)
        out = out.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

if __name__ == '__main__':
    device = torch.device("cuda")
    pose_cnn = PoseCNN(num_input_frames=2).to(device)
    inputs = torch.zeros(8, 6, 192, 640).to(device)
    
    output = pose_cnn(inputs)
    pose_cnn = torch.jit.script(pose_cnn)
    torch.jit.save(pose_cnn, 'pose_cnn.jit.pt')