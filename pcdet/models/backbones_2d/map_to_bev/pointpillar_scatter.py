import torch
import torch.nn as nn
import numpy as np
count = 0

class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1
        self.export_onnx = False

    def forward_export(self, batch_dict, **kwargs):
        # pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        # pillar_features = pillar_features.squeeze(dim=3)
        # pillar_features = pillar_features.squeeze(dim=0)
        # spatial_feature = torch.zeros(
        #     self.num_bev_features,
        #     self.nz * self.nx * self.ny,
        #     dtype=pillar_features.dtype,
        #     device=pillar_features.device)
        # if coords.ndim == 2:
        #     coords = coords[:, 1]
        # spatial_feature[:, coords] = pillar_features
        #
        # batch_spatial_features = spatial_feature.view(1, self.num_bev_features * self.nz, self.ny,self.nx)
        # batch_dict['spatial_features'] = batch_spatial_features
        # return batch_dict

        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        pillar_features = pillar_features.squeeze(dim=3) # (1, C, N)
        spatial_feature = torch.zeros(1,
            self.num_bev_features,
            self.nz * self.nx * self.ny,
            dtype=pillar_features.dtype,
            device=pillar_features.device)

        if coords.ndim == 2:
            coords = coords[:, 1]

        batch_spatial_features = spatial_feature.scatter_(2, coords, pillar_features)
        batch_dict['spatial_features'] = batch_spatial_features.view(1, self.num_bev_features, self.ny, self.nx)
        return batch_dict

    def forward(self, batch_dict, **kwargs):
        if self.export_onnx:
            return self.forward_export(batch_dict, **kwargs)

        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']

        pillar_features = pillar_features.squeeze(dim=3)
        pillar_features = pillar_features.permute(0, 2, 1).contiguous()
        pillar_features = pillar_features.view(-1, pillar_features.size(2)).t()

        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            batch_mask = batch_mask & (coords[:, 1] > 0)
            indices = coords[batch_mask, :][:, 1]
            pillars = pillar_features[:, batch_mask]

            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)

        # global count
        # #
        # import skimage
        # jpg = batch_spatial_features.detach().cpu()[0].abs().sum(0).numpy()
        # print('-----------------------', jpg.min(), jpg.max())
        # jpg = ((jpg - jpg.min()) / (jpg.max() - jpg.min()) * 255 ).astype(np.ubyte)
        # # jpg = jpg.transpose(1,2,0)
        # skimage.io.imsave('./bev_%d.jpg' % count, jpg)
        # count += 1
        batch_dict['spatial_features'] = batch_spatial_features
        # import pdb
        # pdb.set_trace()
        return batch_dict


class NewPointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features = 64
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        self.batch_size = 1
        self.register_buffer("bev_feature",
            torch.zeros(self.batch_size, self.num_bev_features * self.nz, self.nx * self.ny))

    def forward(self, batch_dict):
        pillar_features, coords = batch_dict["pillar_features"], batch_dict["voxel_coords"]
        # coords: shape is (batch_size * num_bev, 4), [batch_index, z, x, y]
        # it should be: (batch_size, num_bev, 3)
        # convert to indices: (batch_size, 1, num_bev)
        # which should be processed in dataloader
        coords = coords.reshape(self.batch_size, -1, 4)
        indices = coords[:, :, 2] + coords[:, :, 3] * self.ny
        indices = indices.reshape(self.batch_size, 1, -1).repeat(1, self.num_bev_features, 1).long()

        pillar_features = pillar_features.t().reshape(1, self.num_bev_features, -1)

        return self.bev_feature.scatter(2, indices, pillar_features).reshape(self.batch_size, self.num_bev_features, self.nx, self.ny)