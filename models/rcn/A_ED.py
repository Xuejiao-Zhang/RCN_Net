import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.rcn.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class AEDModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(AEDModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AED(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        """
        :param backbone: resnet
        :param output_stride: 16
        :param BatchNorm:
        """
        super(AED, self).__init__()

        # AED
        self.aed1_1 = AEDModule(1024, 512, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
        self.aed1_2 = AEDModule(1024, 256, 3, padding=6, dilation=6, BatchNorm=BatchNorm)
        self.aed2_1 = AEDModule(1024, 512, 3, padding=12, dilation=12, BatchNorm=BatchNorm)
        self.aed2_2 = AEDModule(1024, 256, 3, padding=18, dilation=18, BatchNorm=BatchNorm)

        self.conv_small = nn.Sequential(nn.Conv2d(768, 256, 1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5))

        self.conv_big = nn.Sequential(nn.Conv2d(768, 256, 1, bias=False),
                                      BatchNorm(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.conv_all = nn.Sequential(nn.Conv2d(1536, 256, 1, bias=False),
                                      BatchNorm(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self._init_weight()

    def forward(self, x):


        return x_small, x_big, x_all

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aed(backbone, output_stride, BatchNorm):
    return AED(backbone, output_stride, BatchNorm)
