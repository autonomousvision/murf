import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewAgg(nn.Module):
    def __init__(self, feat_ch=128,
                 input_feat_channels=371,
                 ):
        super(MultiViewAgg, self).__init__()
        self.feat_ch = feat_ch
        self.proj = nn.Linear(input_feat_channels, feat_ch)
        self.view_fc = nn.Linear(4, feat_ch)
        in_channels = feat_ch * 3
        self.global_fc = nn.Linear(in_channels, feat_ch)

        self.agg_w_fc = nn.Linear(feat_ch, 1)
        self.fc = nn.Linear(feat_ch, feat_ch)

    def forward(self, img_feat_rgb_dir):
        B, S = len(img_feat_rgb_dir), img_feat_rgb_dir.shape[-2]

        view_feat = self.view_fc(img_feat_rgb_dir[..., -4:])
        img_feat_rgb = self.proj(img_feat_rgb_dir[..., :-4]) + view_feat

        var_feat = torch.var(img_feat_rgb, dim=-2).view(B, -1,
                                                        1, self.feat_ch).repeat(1, 1, S, 1)
        avg_feat = torch.mean(img_feat_rgb, dim=-2).view(B, -1,
                                                         1, self.feat_ch).repeat(1, 1, S, 1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)

        global_feat = self.global_fc(feat)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=-2)
        im_feat = (global_feat * agg_w).sum(dim=-2)
        return self.fc(im_feat)
