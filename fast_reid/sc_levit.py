import timm
import torch.nn as nn


def get_levit(arch_name, pretrained=True, feature_dim=128):
    m = timm.create_model(arch_name, pretrained=pretrained)
    m.head.l = nn.Linear(in_features=768, out_features=feature_dim, bias=True)
    m.head_dist.l = nn.Linear(in_features=768, out_features=feature_dim, bias=True)
    return m