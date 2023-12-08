import torch
import torch.nn as nn

import lpips


# https://github.com/nianticlabs/monodepth2/blob/master/layers.py
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self, patch_size=3):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(patch_size, 1)
        self.mu_y_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_x_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_y_pool = nn.AvgPool2d(patch_size, 1)
        self.sig_xy_pool = nn.AvgPool2d(patch_size, 1)

        self.refl = nn.ReflectionPad2d(patch_size // 2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def lpips_loss_func(pred, gt):
    # image should be RGB, IMPORTANT: normalized to [-1,1]
    assert pred.dim() == 4 and pred.size() == gt.size()  # [B, 3, H, W]
    assert pred.min() >= -1 and pred.max() <= 1 and gt.min() >= -1 and gt.max() <= 1

    loss_func = lpips.LPIPS(net='vgg').to(pred.device)
    for param in loss_func.parameters():
        param.requires_grad = False

    loss = loss_func(pred, gt)

    return loss
