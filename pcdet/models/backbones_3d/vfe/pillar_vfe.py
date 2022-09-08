import pdb

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vfe_template import VFETemplate

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        self.linear = nn.Conv2d(in_channels, out_channels,  kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)

    def forward(self, inputs):
        x = self.linear(inputs)
        x = self.norm(x)
        x = F.relu(x)
        if self.training:
            x_max = x.max(3, keepdim=True)[0]
        else:
            x_max = torch.nn.functional.max_pool2d(x, kernel_size=(1, 4), stride=(1, 4))
            x_max = torch.nn.functional.max_pool2d(x_max, kernel_size=(1, 4), stride=(1, 4))
            x_max = torch.nn.functional.max_pool2d(x_max, kernel_size=(1, 2), stride=(1, 2))
        return x_max

class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        features = batch_dict['voxels']
        for pfn in self.pfn_layers:
            features = pfn(features)
        batch_dict['pillar_features'] = features
        return batch_dict
