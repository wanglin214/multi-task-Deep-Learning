# @Time: 2023/5/31 14:55
# @Author: WangLin
# @File: net.py
# @Software: PyCharm
import os

import torch
from torch import nn
import math
from torch.nn import functional as F
from init_weight import count_param


class Conv_Block(nn.Module):  # Define convolution module - activation function changed from ReLU to LeakyReLU
    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            # nn.Dropout2d(0.3),
            # nn.LeReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),  # When using BatchNorm, bias is not needed
            # nn.Dropout2d(0.3),
            # nn.ReLU()
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):  # Downsampling using convolution to reduce feature loss (max pooling loses more features)
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(  # Downsampling keeps channel count unchanged, convolution with stride=2 halves dimensions
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            # nn.ReLU()
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):  # Upsampling using interpolation to avoid "checkerboard artifacts" from transposed convolution
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)  # Upsampling reduces channel count, but will be concatenated later

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((out, feature_map), dim=1)  # Concatenate operation for skip connections


class LinearBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LinearBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.BatchNorm1d(out_channel),
            # nn.ReLU()
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class multitask(nn.Module):
    def __init__(self):
        super(multitask, self).__init__()
        self.c1 = Conv_Block(1, 64)
        self.d1 = DownSample(64)
        self.c2 = Conv_Block(64, 128)
        self.d2 = DownSample(128)
        self.c3 = Conv_Block(128, 256)
        self.d3 = DownSample(256)
        self.c4 = Conv_Block(256, 512)
        self.d4 = DownSample(512)
        self.c5 = Conv_Block(512, 1024)
        # Second half of 2D convolution regression task 1 (depth)
        self.u1_dep = UpSample(1024)
        self.c6_dep = Conv_Block(1024, 512)
        self.u2_dep = UpSample(512)
        self.c7_dep = Conv_Block(512, 256)
        self.u3_dep = UpSample(256)
        self.c8_dep = Conv_Block(256, 128)
        self.u4_dep = UpSample(128)
        self.c9_dep = Conv_Block(128, 64)
        self.c10_dep = nn.Conv2d(64, 1, 1, 1, 0, bias=False)
        # self.out_dep = nn.LeakyReLU()
        # # Second half of 2D convolution regression task 2 (gravity)
        # self.u1_grav = UpSample(256)
        # self.c6_grav = Conv_Block(256, 128)
        # self.u2_grav = UpSample(128)
        # self.c7_grav = Conv_Block(128, 64)
        # self.u3_grav = UpSample(64)
        # self.c8_grav = Conv_Block(64, 32)
        # self.u4_grav = UpSample(32)
        # self.c9_grav = Conv_Block(32, 16)
        # self.c10_grav = nn.Conv2d(16, 1, 1, 1, 0, bias=False)
        # self.out_grav = nn.ReLU()
        # Second half of 0D density regression task
        self.c6linear = nn.Conv2d(1024, 256, 1, 1, 0, bias=False)
        self.fc1 = LinearBlock(4096, 1024)
        self.fc2 = LinearBlock(1024, 256)
        self.fc3 = LinearBlock(256, 64)
        self.fc4 = LinearBlock(64, 16)
        self.fc5 = LinearBlock(16, 4)
        self.fc6 = nn.Linear(4,1)
        # self.out_dens  = nn.LeakyReLU()

        # Kaiming weight initialization
        self._init_weights()

    # see also https://github.com/pytorch/pytorch/issues/18182
    def _init_weights(self):  # He (Kaiming) weight initialization method
        for m in self.modules():
            if type(m) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, x):
        # Common downsampling part (encoder)
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        # 2D depth regression task 1 output (decoder with skip connections)
        O1_dep = self.c6_dep(self.u1_dep(R5, R4))
        O2_dep = self.c7_dep(self.u2_dep(O1_dep, R3))
        O3_dep = self.c8_dep(self.u3_dep(O2_dep, R2))
        # print(O3_dep.shape, R2.shape, R1.shape)
        # os.system('pause')
        O4_dep = self.c9_dep(self.u4_dep(O3_dep, R1))
        # print(O3_dep.shape, R1.shape)
        # os.system('pause')
        O5_dep = self.c10_dep(O4_dep)
        depth = O5_dep
        # 0D density regression task 2 (fully connected layers)
        R6 = self.c6linear(R5)  # R5[256,4,4]
        x = R6.view(R6.size(0), -1)  # Flatten 2D tensor to 1D, becoming [4096,1]
        # print(x.shape)
        O1_den = self.fc1(x)
        # print(O1.shape)
        O2_den = self.fc2(O1_den)
        O3_den = self.fc3(O2_den)
        O4_den = self.fc4(O3_den)
        O5_den = self.fc5(O4_den)
        O6_den = self.fc6(O5_den)
        density = O6_den
        # 2D gravity anomaly regression task 3 output
        # O1_grav = self.c6_grav(self.u1_grav(R5, R4))
        # O2_grav = self.c7_grav(self.u2_grav(O1_grav, R3))
        # O3_grav = self.c8_grav(self.u3_grav(O2_grav, R2))
        # O4_grav = self.c9_grav(self.u4_grav(O3_grav, R1))
        # O5_grav = self.c10_grav(O4_grav)
        # gravity = self.out_grav(O5_grav)

        return depth, density  # , gravity


if __name__ == '__main__':
    x = torch.randn(4, 1, 64, 64)
    net = multitask()
    print(count_param(net))  # Calculate number of network parameters
    depth, density = net(x)

    print(depth.shape, density.shape)

    # print(net)
    # print(net.state_dict())  # View initial network weights
