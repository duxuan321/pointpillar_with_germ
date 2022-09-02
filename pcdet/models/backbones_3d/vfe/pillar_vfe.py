import pdb

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# state = torch.load('/home/yuanxin/mvlidarnet_pcdet/pointpillar_7728.pth')
# for k, v in state['model_state'].items():
#     print(k, v.size())
# 1/0
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

        if self.use_norm:
            # self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.linear = nn.Conv2d(in_channels, out_channels,  kernel_size=1, stride=1,padding=0,bias=False)
            # self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):

        x = self.linear(inputs)
        # x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        x = self.norm(x)
        # import pdb
        # pdb.set_trace()
        # print(self.norm)

        x = F.relu(x)
        x_max = torch.nn.functional.max_pool2d(x, kernel_size=(1, 4), stride=(1, 4))
        x_max = torch.nn.functional.max_pool2d(x_max, kernel_size=(1, 4), stride=(1, 4))
        x_max = torch.nn.functional.max_pool2d(x_max, kernel_size=(1, 2), stride=(1, 2))
        # print(x_max.shape)
        # import pdb
        # pdb.set_trace()
        # x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        # else:
        #     x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        #     x_concatenated = torch.cat([x, x_repeat], dim=2)
        #     return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        # self.use_norm = self.model_cfg.USE_NORM
        self.use_norm = True
        # self.with_distance = self.model_cfg.WITH_DISTANCE
        self.with_distance = False
        # self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        self.use_absolute_xyz = True
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        # self.num_filters = self.model_cfg.NUM_FILTERS
        self.num_filters = [64]
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

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.fake_quant = False
        self.weight = torch.Tensor([1/70, 1/80, 1/4, 1.0, 1/0.32, 1/0.32, 1/8, 1/0.16, 1/0.16, 1/4]).cuda().view(1,1,-1)
        self.bias = torch.Tensor([-0.5, 0.0, 0.25, -0.5, 0, 0, 0, 0, 0, 0]).cuda().view(1,1,-1)

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward_fake_quant(self, features, **kwargs):
        features = features[0]
        for pfn in self.pfn_layers:
            features = pfn(features)
        return features

    def forward(self, batch_dict, **kwargs):
        if self.fake_quant:
            return self.forward_fake_quant(batch_dict)

    # def forward(self, features, **kwargs):
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        export_onnx = False

        if export_onnx:
        # 导出pfn部分的onnx时打开,batch_size为1
            maxnum = 8694
            temp = torch.zeros([maxnum - voxel_features.shape[0], voxel_features.shape[1], voxel_features.shape[2]]).cuda()
            voxel_features = torch.cat((voxel_features, temp), dim=0)
            temp = torch.zeros([maxnum - voxel_num_points.shape[0]]).cuda()
            voxel_num_points = torch.cat((voxel_num_points, temp), dim=0)
            temp = torch.zeros([maxnum - coords.shape[0], coords.shape[1]]).cuda()
            coords = torch.cat((coords, temp), dim=0)
            batch_dict['voxels'] = voxel_features
            batch_dict['voxel_num_points'] = voxel_num_points
            batch_dict['voxel_coords'] = coords
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / (voxel_num_points.type_as(voxel_features).view(-1, 1, 1)+0.00001)
        f_cluster = voxel_features[:, :, :3] - points_mean

        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

        features = features * self.weight + self.bias
        features *= mask

        # print(features.view(-1, 10).min(0)[0], features.view(-1, 10).max(0)[0])
        features = features.permute(2,0,1)
        features = features.unsqueeze(dim=0)

        for pfn in self.pfn_layers:
            features = pfn(features)

        # 导出pfn部分的onnx
        import io
        import onnx



        if export_onnx:
            pfn.eval()
            input_shape = (1, 10, maxnum, 32)
            buffer = io.BytesIO()
            torch.onnx.export(pfn.cuda(),torch.randn(input_shape).cuda(),buffer,opset_version=11,training=False)
            onnx_model = onnx.load_from_string(buffer.getvalue())
            onnx.save(onnx_model ,'/home/yuanxin/mvlidarnet_pcdet/pfn.onnx')
        #
        features = features.squeeze(dim=0)
        features = features.permute(1, 2, 0)

        features = features.squeeze()
        # features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict
