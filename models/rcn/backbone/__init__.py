from models.rcn.backbone import RCNNet


def build_backbone(backbone, output_stride, BatchNorm, device):

    if backbone == 'RCNNet':
        return RCNNet.RCNNet(output_stride, device, BatchNorm)
    else:
        raise NotImplementedError
