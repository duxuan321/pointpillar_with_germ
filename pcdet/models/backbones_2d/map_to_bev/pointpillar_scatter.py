import torch
import torch.nn as nn
import numpy as np
count = 0
class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        # self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.num_bev_features = 64
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1


    def forward(self, batch_dict, **kwargs):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nz * self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        batch_spatial_features = batch_spatial_features.permute(0,1,3,2).contiguous()
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
