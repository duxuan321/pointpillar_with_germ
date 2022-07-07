import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=stride, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()]
    return nn.Sequential(*layers)


def upconv(in_channels, out_channels):
    layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()]
    return nn.Sequential(*layers)


class MVLidarNetBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        multi_input_channels = self.model_cfg.get('MULTI_INPUT_CHANNELS', [3, 7])
        """
        self.sem = nn.Sequential(conv3x3(multi_input_channels[1], 16),
                                 conv3x3(16, 16),
                                 conv3x3(16, 32, 2),
                                 conv3x3(32, 32))
        """
        self.height = nn.Sequential(conv3x3(multi_input_channels[0], 16),
                                 conv3x3(16, 16),
                                 conv3x3(16, 32, 2),
                                 conv3x3(32, 64))   # 这里改成了32 64，因为暂时没有分割信息
        
        self.block1a = conv3x3(64, 64)
        self.block1b = conv3x3(64, 64, 2)
        self.block2a = conv3x3(64, 128)
        self.block2b = conv3x3(128, 128, 2)
        self.block3a = conv3x3(128, 256)
        self.block3b = conv3x3(256, 256, 2)
        
        self.up1a = upconv(256, 128)
        self.up1c = conv3x3(256, 128)
        
        self.up2a = upconv(128, 64)
        self.up2c = conv3x3(128, 64)

        self.num_bev_features = 64

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        height_feat = data_dict['spatial_features']
        # import skimage
        # jpg = height_feat.detach().cpu().squeeze().numpy()
        # print('-----------------------', jpg.min(), jpg.max())
        # jpg = ((jpg - jpg.min()) / (jpg.max() - jpg.min()) * 255 ).astype(np.ubyte)
        # jpg = jpg.transpose(1,2,0)
        # skimage.io.imsave('./bev.jpg', jpg)

        height_feat = self.height(height_feat)
        
        f_block1a = self.block1a(height_feat)
        f_block1b = self.block1b(f_block1a)
        f_block2a = self.block2a(f_block1b)
        f_block2b = self.block2b(f_block2a)
        f_block3a = self.block3a(f_block2b)
        f_block3b = self.block3b(f_block3a)
        

        f_up1a = self.up1a(f_block3b)
        f_up1b = torch.cat([f_up1a, f_block2b], 1)
        f_up1c = self.up1c(f_up1b)
        
        f_up2a = self.up2a(f_up1c)
        f_up2b = torch.cat([f_up2a, f_block1b], 1)
        f_up2c = self.up2c(f_up2b)

        data_dict['spatial_features_2d'] = f_up2c

        return data_dict
