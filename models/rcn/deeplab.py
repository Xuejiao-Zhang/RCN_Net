import torch
import torch.nn as nn
import torch.nn.functional as F
from models.rcn.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.rcn.A_ED import build_aed
from models.rcn.grasp import build_grasp
from models.rcn.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, angle_cls, device, backbone='RCNNet', output_stride=16, num_classes=8, sync_bn=False,
                 freeze_bn=False, size=320):
        super(DeepLab, self).__init__()
        BatchNorm = nn.BatchNorm2d
        print("*******************************************************")
        print(backbone)
        self.backbone = build_backbone(backbone, output_stride, BatchNorm, device)      # 主干网络
        self.aed = build_aed(backbone, output_stride, BatchNorm)              # A_ED
        self.grasp = build_grasp(num_classes, backbone, BatchNorm, size, angle_cls=angle_cls)          # 解码器

        self.freeze_bn = freeze_bn




    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aed, self.grasp]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
