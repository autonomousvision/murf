import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .ldm_unet.unet import UNetModel
from .nerf import NeRF
from .utils import MultiViewAgg


class CondNeRFFine(NeRF):
    def __init__(self,
                 net_width=16,
                 num_samples=16,
                 unet_num_res_blocks=1,
                 device='cuda',
                 **kwargs,
                 ):

        self.device = device

        self.net_width = net_width
        self.num_samples = num_samples
        self.unet_num_res_blocks = unet_num_res_blocks

        self.cos_n_group = [1, 1, 1, 1]

        super(CondNeRFFine, self).__init__(None)

    def define_network(self, opt):

        W = self.net_width
        color_channels = 3

        proj_channels = 8
        # project features to lower dim
        self.feature_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(128, 128, 1), nn.GELU(),
                          nn.Conv2d(128, proj_channels, 1)),
            nn.Sequential(nn.Conv2d(128, 128, 1), nn.GELU(),
                          nn.Conv2d(128, proj_channels, 1)),
            nn.Sequential(nn.Conv2d(96, 96, 1), nn.GELU(),
                          nn.Conv2d(96, proj_channels, 1)),
            nn.Sequential(nn.Conv2d(64, 64, 1), nn.GELU(),
                          nn.Conv2d(64, proj_channels, 1))
        ])
        sampled_feature_channels = proj_channels * 4
        input_channels = sampled_feature_channels + 3  # feature and color

        # multi-view color and feature aggregation
        self.mvagg = MultiViewAgg(feat_ch=self.net_width,
                                  input_feat_channels=input_channels
                                  )

        input_ch_feat = sum(self.cos_n_group)

        input_ch_feat += color_channels

        # (cosine, feature, color)
        input_ch_feat = self.net_width + sum(self.cos_n_group)

        # residual_color_cosine
        self.residual = nn.Conv3d(input_ch_feat, W, 1, 1, 0)

        self.regressor = UNetModel(image_size=None,
                                   in_channels=input_ch_feat,
                                   model_channels=W,
                                   out_channels=W,
                                   num_res_blocks=self.unet_num_res_blocks,
                                   attention_resolutions=[],
                                   channel_mult=[1, 1, 2, 4],
                                   num_head_channels=8,
                                   dims=3,
                                   postnorm=True,
                                   channels_per_group=4,
                                   condition_channels=input_ch_feat,
                                   )

        channels = W
        self.density_head = nn.Sequential(nn.Linear(channels, channels),
                                          nn.GELU(),
                                          nn.Linear(channels, 1),
                                          nn.Softplus()
                                          )

        self.rgb_head = nn.Sequential(nn.Linear(channels, channels),
                                      nn.GELU(),
                                      nn.Linear(channels, 3),
                                      nn.Sigmoid(),
                                      )

    def forward(self, cond_info=None, img_hw=None,
                **kwargs,
                ):

        assert img_hw is not None
        curr_h, curr_w = img_hw

        # construct input to the decoder
        b_, l_, d_ = cond_info['color_info'].shape[:3]

        colors = cond_info['color_info'].permute(
            0, 1, 2, 4, 3).contiguous()   # [B, H*W, D, V, (2R+1)^2 * 3]
        v_colors = colors.size(-2)

        viewdir_diff = cond_info['viewdir_diff']  # [B, V, H*W, D, 4]

        viewdir_diff = viewdir_diff.permute(0, 2, 3, 1, 4)  # [B, H*W, D, V, 4]

        features = cond_info['sampled_feature_info']  # [B, V, C, H*W, D]
        b_, v_features, _, l_, d_ = features.shape
        features = features.permute(
            0, 3, 4, 1, 2).contiguous()  # [B, H*W, D, V, C]

        concat = torch.cat((features, colors, viewdir_diff),
                           dim=-1).view(b_, l_ * d_, v_colors, -1)  # [B, H*W*D, V, C]

        agg = self.mvagg(concat)  # [B, H*W*D, C]

        agg = agg.view(b_, l_, d_, -1)  # [B, H*W, D, C]

        agg_input = torch.cat(
            (cond_info['feat_info'], agg), dim=-1)  # [B, H*W, D, C]

        # decoder
        batch_size, _, n_samples, _ = agg_input.shape

        conv_input = agg_input  # [B, H*W, D, C]

        conv_input = conv_input.reshape(
            batch_size, curr_h, curr_w, n_samples, -1).permute(0, 4, 1, 2, 3).contiguous()  # [B, C, H, W, D]

        # add residual connection from input to the head
        residual_color_cosine = self.residual(
            conv_input)  # [B, 64, H, W, D]

        # [B, C, D, H, W] for ldmunet, since downsampling is only performed in H, W dims
        x_ = conv_input.permute(0, 1, 4, 2, 3)
        x_ = self.regressor(x_)
        out = x_.permute(0, 1, 3, 4, 2)  # [B, C, H, W, D]

        # add residual connection from input to the head
        out = out + residual_color_cosine

        h = out.reshape(batch_size, -1, curr_h * curr_w,
                        n_samples).permute(0, 2, 3, 1)  # [B, H*W, D, C]

        # output head
        density = self.density_head(h).squeeze(-1)  # [B, H*W, D]
        rgb = self.rgb_head(h)  # [B, H*W, D, 3]

        return rgb, density
