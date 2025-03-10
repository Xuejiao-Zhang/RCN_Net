import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.rcn.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Grasp(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, upSize, angle_cls):
        """
        :param num_classes:
        :param backbone:
        :param BatchNorm:
        :param upSize: 320
        """
        super(Grasp, self).__init__()

        self.upSize = upSize
        self.angleLabel = angle_cls

        # feat_low 卷积
        self.conv_1 = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                    BatchNorm(48),
                                    nn.ReLU())

        self.conv_2 = nn.Sequential(nn.Conv2d(512, 48, 1, bias=False),
                                    BatchNorm(48),
                                    nn.ReLU())

        # aed_small 卷积
        self.conv_aed_small = nn.Sequential(nn.Conv2d(560, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      BatchNorm(256),
                                      nn.ReLU())

        # aed_mid 卷积
        self.conv_aed_mid = nn.Sequential(nn.Conv2d(352, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())

        # aed_big 卷积
        self.conv_aed_big = nn.Sequential(nn.Conv2d(304, 304, kernel_size=3, stride=1, padding=1, bias=False),
                                           BatchNorm(304),
                                           nn.ReLU())

        # 抓取置信度预测
        self.able_conv = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),

                                       nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),

                                       nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                       nn.ReLU(),

                                       nn.Conv2d(128, 1, kernel_size=1, stride=1))

        # 角度预测
        self.angle_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(256, self.angleLabel, kernel_size=1, stride=1))

        # 抓取宽度预测
        self.width_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(128),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(128, 1, kernel_size=1, stride=1))

        self._init_weight()


    def forward(self, feat_1, aed_small, aed_big, aed_all):
        """
        :param feat_low: Res_1 的输出特征            (-1, 256, 120, 160)
        :param aed_small: rate = {1, 6}            (-1, 256, 30, 40)
        :param aed_big: rate = {12, 18}            (-1, 256, 30, 40)
        :param aed_all: rate = {1, 6, 12, 18}      (-1, 256, 30, 40)
        """
        # feat_1 卷积
        feat_1 = self.conv_1(feat_1)

        # 特征融合
        aed_small = F.interpolate(aed_small, size=feat_1.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        aed_small = torch.cat((aed_small, feat_1), dim=1)
        aed_big = F.interpolate(aed_big, size=feat_1.size()[2:], mode='bilinear', align_corners=True)   # 上采样（双线性插值）
        aed_small = torch.cat((aed_small, aed_big), dim=1)
        input_able = self.conv_aed_small(aed_small)

        # angle width 获取输入

        aed_all = F.interpolate(aed_all, size=feat_1.size()[2:], mode='bilinear', align_corners=True)  # 上采样（双线性插值）
        aed_all = torch.cat((aed_all, feat_1), dim=1)
        aed_all = self.conv_aed_big(aed_all)
        # 预测
        able_pred = self.able_conv(input_able)
        angle_pred = self.angle_conv(aed_all)
        width_pred = self.width_conv(aed_all)

        return able_pred, angle_pred, width_pred

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_grasp(num_classes, backbone, BatchNorm, upSize, angle_cls):
    return Grasp(num_classes, backbone, BatchNorm, upSize, angle_cls)
