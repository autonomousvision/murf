import torch
import os
import torch.nn.functional as F
import torch.nn as nn
from math import log2
from .nerf import NeRF
from .resblock import BasicBlock
from .utils import MultiViewAgg


class CondNeRF(NeRF):
    def __init__(self, opt,
                 ):
        super(CondNeRF, self).__init__(opt)

    def define_network(self, opt):

        W = opt.decoder.net_width

        if getattr(opt, 'weighted_cosine', False):
            self.vis = nn.Sequential(
                nn.Conv2d(1, 8, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(8, 8, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(8, 1, 3, 1, 1),
                nn.Sigmoid(),
            )

        # merge multi-view feature, color and view dir diff information with predicted weights
        sampled_feature_channels = 64 + 96 + 2 * 128  # cnn and transformer features
        input_channels = sampled_feature_channels + 3 * \
            (2 * getattr(opt, 'sample_color_window_radius', 0) + 1) ** 2

        opt.feature_agg_channel = opt.decoder.net_width
        self.mvagg = MultiViewAgg(feat_ch=opt.feature_agg_channel,
                                  input_feat_channels=input_channels,
                                  )

        input_ch_feat = opt.feature_agg_channel + sum(opt.encoder.cos_n_group)

        # residual connection
        self.residual = nn.Conv3d(input_ch_feat, W, 1, 1, 0)

        # decoder
        modules = [nn.Conv3d(input_ch_feat, W, (3, 3, 1), 1, (1, 1, 0)),
                   nn.LeakyReLU(0.1),
                   nn.Conv3d(W, W, (1, 1, 3), 1, (0, 0, 1)),
                   nn.GroupNorm(8, W),
                   nn.LeakyReLU(0.1),
                   ]

        decoder_num_resblocks = opt.decoder_num_resblocks

        for i in range(decoder_num_resblocks):
            modules.append(BasicBlock(W, W, kernel=3,
                                      conv_2plus1d=True,
                                      )
                           )

        self.regressor = nn.Sequential(*modules)

        # output head
        channels = opt.upconv_channel_list[-1]

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

        # upsample
        upsample_factor = opt.radiance_subsample_factor
        channels = W

        # upsampler
        num_upsamples = int(log2(upsample_factor))
        self.up_convs = nn.ModuleList()
        for i in range(num_upsamples):
            channel_expansion = 4

            # specify the upsampling conv channels in list
            upconv_channel_list = opt.upconv_channel_list
            assert len(upconv_channel_list) == num_upsamples

            self.up_convs.append(nn.Conv3d(
                upconv_channel_list[i], upconv_channel_list[i + 1 if i < num_upsamples - 1 else i] * channel_expansion, 3, 1, 1))

        self.pixel_shuffle = nn.PixelShuffle(2)
        # conv after the final pixelshuffle layer
        self.conv = nn.Conv3d(
            upconv_channel_list[-1], upconv_channel_list[-1], 3, 1, 1)

    def forward(self, opt, cond_info=None,
                img_hw=None,
                num_views=None,
                **kwargs,
                ):

        opt.n_src_views = num_views

        assert img_hw is not None
        curr_h, curr_w = img_hw

        if self.training and getattr(opt, 'random_crop', False):
            curr_h, curr_w = opt.crop_height, opt.crop_width

        if getattr(opt, 'radiance_subsample_factor', False):
            curr_h = curr_h // opt.radiance_subsample_factor
            curr_w = curr_w // opt.radiance_subsample_factor

        if getattr(opt, 'weighted_cosine', False):
            cosine = cond_info['feat_info']  # [B, V, D, H*W, C]
            # predict the visibility based on entropy
            cosine_sum = cosine.sum(dim=-1)  # [B, V, D, H*W]
            cosine_sum_norm = F.softmax(
                cosine_sum.detach(), dim=2)  # [B, V, D, H*W]
            entropy = (-cosine_sum_norm * torch.log(cosine_sum_norm +
                       1e-6)).sum(dim=2, keepdim=True)  # [B, V, 1, H*W]

            b_, v_ = entropy.shape[:2]

            vis_weight = self.vis(entropy.view(
                b_ * v_, 1, curr_h, curr_w)).view(b_, v_, 1, -1, 1)  # [B, V, 1, H*W, 1]

            merged_cosine = (cosine * vis_weight).sum(dim=1) / \
                (vis_weight.sum(1) + 1e-6)  # [B, D, H*W, C]

            cond_info['feat_info'] = merged_cosine.permute(
                0, 2, 1, 3)  # [B, H*W, D, C]

        # multi-view aggregation of features and colors
        features = cond_info['sampled_feature_info']  # [B, V, C, H*W, D]
        b_, v_, _, l_, d_ = features.shape
        features = features.permute(
            0, 3, 4, 1, 2).contiguous()  # [B, H*W, D, V, C]

        colors = cond_info['color_info'].permute(
            0, 1, 2, 4, 3).contiguous()   # [B, H*W, D, V, (2R+1)^2 * 3]
        v_ = colors.size(-2)

        viewdir_diff = cond_info['viewdir_diff']  # [B, V, H*W, D, 4]

        viewdir_diff = viewdir_diff.permute(0, 2, 3, 1, 4)  # [B, H*W, D, V, 4]

        # 128+128+96+64, (2*8+1)**2 * 3, 4
        concat = torch.cat((features, colors, viewdir_diff),
                           dim=-1).view(b_, l_ * d_, v_, -1)  # [B, H*W*D, V, C]
        agg = self.mvagg(concat)  # [B, H*W*D, C]

        agg = agg.view(b_, l_, d_, -1)  # [B, H*W, D, C]

        # decoder input
        conv_input = torch.cat(
            (cond_info['feat_info'], agg), dim=-1)  # [B, H*W, D, C]

        batch_size, _, n_samples, _ = conv_input.shape

        conv_input = conv_input.reshape(
            batch_size, curr_h, curr_w, n_samples, -1).permute(0, 4, 1, 2, 3).contiguous()  # [B, C, H, W, D]

        # add residual connection from input to the head
        residual_color_cosine = self.residual(
            conv_input)  # [B, 64, H, W, D]

        out = self.regressor(conv_input)  # [B, C, H, W, D]

        # add residual connection from input to the head
        out = out + residual_color_cosine

        # upsample
        for i in range(len(self.up_convs)):

            # out: [B, C, H, W, D]
            out = self.up_convs[i](out)

            # pixel shuffle upsampling
            # [B, D, C, H, W]
            out = out.permute(0, 4, 1, 2, 3)
            out = self.pixel_shuffle(out)
            out = F.leaky_relu(out, 0.1, inplace=True)
            # [B, C, H, W, D]
            out = out.permute(0, 2, 3, 4, 1)

            if i + 1 == len(self.up_convs):
                # conv at the final resolution
                out = self.conv(out)

        upsample_factor = opt.radiance_subsample_factor

        h = out.reshape(batch_size, -1, curr_h * curr_w * (upsample_factor ** 2),
                        n_samples).permute(0, 2, 3, 1)  # [B, H*W, D, C]

        # output head
        density = self.density_head(h).squeeze(-1)  # [B, H*W, D]
        rgb = self.rgb_head(h)  # [B, H*W, D, 3]

        return rgb, density
