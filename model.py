import torch
from torch import nn
import torch.hub
from torchvision.models.segmentation.deeplabv3 import (
    DeepLabHead, DeepLabV3)
from torchvision.models.segmentation.fcn import (FCNHead, FCN)


class EfficientNetFeatureMapGetter(nn.Module):
    def __init__(self, model, feature_map_name='reduction_5'):
        super().__init__()
        self.model = model
        self.feature_map_name = feature_map_name

    def forward(self, x):
        feature_maps = self.model.extract_endpoints(x)
        out_dict = {'out': feature_maps[self.feature_map_name]}
        return out_dict


def load_efficientnet(name, num_classes=1000, pretrained='imagenet'):
    model = torch.hub.load(
        'lukemelas/EfficientNet-PyTorch',
        name,
        num_classes=num_classes,
        pretrained=pretrained
    )
    return model


def make_segmentation_model(name,
                            backbone_name,
                            num_classes,
                            pretrained_backbone='imagenet'):
    """ Factory method. Adapted from
    https://github.com/pytorch/vision/blob/9e7a4b19e3927e0a6d6e237d7043ba904af4682e/torchvision/models/segmentation/segmentation.py
    """

    backbone_name = backbone_name.lower()

    effnet = load_efficientnet(backbone_name, num_classes, pretrained_backbone)

    backbone = EfficientNetFeatureMapGetter(
        effnet, feature_map_name='reduction_5')

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = {
        'efficientnet_b0': 1280,
        'efficientnet_b1': 1280,
        'efficientnet_b2': 1280 + 128,
        'efficientnet_b3': 1280 + 256,
        'efficientnet_b4': 1280 + 512,
        'efficientnet_b5': 2048,
        'efficientnet_b6': 2048 + 256,
        'efficientnet_b7': 2048 + 512,
    }
    classifier = model_map[name][0](inplanes[backbone_name], num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier)
    return model
