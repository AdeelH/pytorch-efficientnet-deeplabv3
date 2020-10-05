import math
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torch.hub
from torchvision.models.segmentation.deeplabv3 import (
    DeepLabHead, DeepLabV3)
from torchvision.models.segmentation.fcn import (FCNHead, FCN)


class EfficientNetFeatureMapGetter(nn.Module):
    def __init__(self, model, feature_map_name='reduction_5', scale_factor=1):
        super().__init__()
        self.model = model
        self.feature_map_name = feature_map_name
        self.scale_factor = scale_factor

    def forward(self, x):
        feature_maps = self.model.extract_endpoints(x)
        feature_map = feature_maps[self.feature_map_name]
        if self.scale_factor > 1:
            feature_map = F.interpolate(
                feature_map, scale_factor=self.scale_factor)
        out_dict = {'out': feature_map}
        return out_dict


class EfficientNetFPN(nn.Module):
    def __init__(self, model, d=256, last_n=None):
        super().__init__()
        self.model = model

        feature_map_shapes = _get_shapes(model)[-last_n:]
        if last_n is not None:
            feature_map_shapes = feature_map_shapes[-last_n:]
        feature_map_shapes = feature_map_shapes[::-1]

        feature_map_channels = [s[1] for s in feature_map_shapes]
        self.convs = nn.ModuleList([
            nn.Conv2d(c, d, kernel_size=(1, 1)) for c in feature_map_channels
        ])

        feature_map_sizes = [s[-2:] for s in feature_map_shapes[1:]]
        self.upsamplers = [
            partial(
                F.interpolate, size=size, mode='bilinear', align_corners=True)
            for size in feature_map_sizes]

    def top_down(self, feature_maps):
        feature_maps = feature_maps[::-1]
        out = self.convs[0](feature_maps[0])
        for f, m, up in zip(feature_maps[1:], self.convs[1:], self.upsamplers):
            out = m(f) + up(out)
        return out

    def forward(self, x):
        feature_maps = self.model.extract_endpoints(x)
        out = self.top_down(list(feature_maps.values()))
        out_dict = {'out': out}
        return out_dict


def _load_efficientnet(name,
                       num_classes=1000,
                       pretrained='imagenet',
                       in_channels=3):
    model = torch.hub.load(
        'lukemelas/EfficientNet-PyTorch',
        name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels
    )
    return model


def _get_shapes(m):
    state = m.training
    m.eval()
    with torch.no_grad():
        feats = m.extract_endpoints(torch.empty(1, 3, 224, 224))
    m.train(state)
    return [f.shape for f in feats.values()]


def _get_inplanes(m, feature_map_name=None):
    state = m.training
    m.eval()
    with torch.no_grad():
        feats = m.extract_endpoints(torch.empty(1, 3, 300, 300))
    m.train(state)
    if feature_map_name is not None:
        return feats[feature_map_name].shape[1]
    return [f.shape[1] for f in feats.values()]


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 image_size=None,
                 **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         **kwargs)
        self.stride = self.stride if len(
            self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size,
                                                        int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max(
            (oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih,
            0)
        pad_w = max(
            (ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw,
            0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w - pad_w // 2, pad_w - pad_w // 2, pad_h - pad_h // 2,
                 pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                     self.dilation, self.groups)
        return x


def make_segmentation_model(name,
                            backbone_name,
                            num_classes,
                            in_channels=3,
                            scale_factor=1,
                            pretrained_backbone='imagenet'):
    """ Factory method. Adapted from
    https://github.com/pytorch/vision/blob/9e7a4b19e3927e0a6d6e237d7043ba904af4682e/torchvision/models/segmentation/segmentation.py
    """

    backbone_name = backbone_name.lower()

    effnet = _load_efficientnet(
        name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained_backbone,
        in_channels=in_channels
    )
    i = 0
    for b in effnet._blocks:
        if b._depthwise_conv.stride[0] == 2:
            if i > 1:
                args = {
                    'kernel_size': b._depthwise_conv.kernel_size,
                    'bias': b._depthwise_conv.bias,
                    'padding': b._depthwise_conv.padding,
                    'groups': b._depthwise_conv.groups
                }
                b._depthwise_conv = Conv2dStaticSamePadding(
                    b._depthwise_conv.in_channels,
                    b._depthwise_conv.out_channels,
                    stride=1,
                    dilation=2,
                    image_size=300,
                    **args)
            i += 1

    backbone = EfficientNetFeatureMapGetter(
        effnet, feature_map_name='reduction_3', scale_factor=scale_factor)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = _get_inplanes(effnet, 'reduction_3')
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier)
    return model


def make_segmentation_model_fpn(name,
                                backbone_name,
                                num_classes,
                                in_channels=3,
                                pretrained_backbone='imagenet',
                                fpn_channels=None,
                                last_n=3):
    """ Factory method. Adapted from
    https://github.com/pytorch/vision/blob/9e7a4b19e3927e0a6d6e237d7043ba904af4682e/torchvision/models/segmentation/segmentation.py
    """

    backbone_name = backbone_name.lower()

    effnet = _load_efficientnet(
        name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained_backbone,
        in_channels=in_channels
    )

    backbone = EfficientNetFPN(effnet, last_n=last_n, d=fpn_channels)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }

    classifier = model_map[name][0](fpn_channels, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier)
    return model
