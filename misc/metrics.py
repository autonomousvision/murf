import numpy as np
from collections import OrderedDict
from skimage.metrics import structural_similarity
import torch
import lpips

from misc.utils import suppress


class EvalTools(object):
    """docstring for EvalTools."""
    def __init__(self, device):
        super(EvalTools, self).__init__()
        self.support_metrics = ['PSNR', 'SSIM', 'LPIPS', 'depth_abs', 'thres005', 'thres001']
        self.device = device
        with torch.no_grad(), suppress(stdout=True, stderr=True):
            self.lpips_metric = lpips.LPIPS(net='vgg').to(device)

    def set_inputs(self, pred_img, gt_img, img_mask=None, full_img_eval=False,
                   pred_depth=None,
                   gt_depth=None,
                   ):
        self.full_pred = pred_img
        self.full_gt = gt_img

        if full_img_eval:
            # print('full image eval')
            self.img_mask = None
            self.proc_pred = pred_img
            self.proc_gt = gt_img
        else:
            if img_mask is not None:
                # print('evaluation with depth mask: ', np.mean(img_mask))  # ~30% mask
                self.img_mask = img_mask
                self.proc_pred = pred_img.copy()
                self.proc_gt = gt_img.copy()
                self.proc_pred[img_mask] = 0.
                self.proc_gt[img_mask] = 0.
            else:  # center crop to 80%
                # print('center crop evaluation')
                # TODO: use full image evaluation
                # print('center crop eval')
                self.img_mask = None
                H_crop, W_crop = np.array(pred_img.shape[:2]) // 10
                self.proc_pred = pred_img[H_crop:-H_crop, W_crop:-W_crop]
                self.proc_gt = gt_img[H_crop:-H_crop, W_crop:-W_crop]

        if pred_depth is not None:
            self.pred_depth = pred_depth
            self.gt_depth = gt_depth
        else:
            self.pred_depth = None
            self.gt_depth = None
        
    def get_psnr(self, pred_img, gt_img, use_mask):
        if use_mask:
            mse = np.mean((pred_img[~self.img_mask] - gt_img[~self.img_mask]) ** 2)
        else:
            mse = np.mean((pred_img - gt_img) ** 2)
        psnr = -10. * np.log(mse) / np.log(10.)
        return psnr

    def get_ssim(self, pred_img, gt_img, **kwargs):
        ssim = structural_similarity(pred_img, gt_img, channel_axis=-1)
        return ssim

    @torch.no_grad()
    def get_lpips(self, pred_img, gt_img, **kwargs):
        pred_tensor = torch.from_numpy(pred_img)[None].permute(0,3,1,2).float() * 2 - 1.0 # image should be RGB, IMPORTANT: normalized to [-1,1]
        gt_tensor = torch.from_numpy(gt_img)[None].permute(0,3,1,2).float() * 2 - 1.0
        lpips = self.lpips_metric(pred_tensor.to(self.device), gt_tensor.to(self.device))
        return lpips.item()

    def get_metrics(self, metrics=None, return_full=False):
        out_dict = OrderedDict()
        if metrics is None:
            metrics = self.support_metrics
        for metric in metrics:
            assert metric in self.support_metrics, f"only support metrics: [{','.join(self.support_metrics)}]"
            if metric == 'depth_abs' or metric == 'thres005' or metric == 'thres001':
                continue

            eval_func = getattr(self, f'get_{metric.lower()}')
            out_dict[metric] = eval_func(self.proc_pred, self.proc_gt, use_mask=(self.img_mask is not None))
            if return_full:
                out_dict[f'{metric}_Full'] = eval_func(self.full_pred, self.full_gt, use_mask=False)

        # additional depth error
        if self.pred_depth is not None and self.gt_depth is not None:
            valid_mask = self.gt_depth != 0.
            depth_abs_error = np.mean(np.abs((self.pred_depth[valid_mask] - self.gt_depth[valid_mask])))
            out_dict['depth_abs'] = depth_abs_error

            out_dict['thres005'] = acc_threshold(self.pred_depth, self.gt_depth, valid_mask, threshold=0.05)

            out_dict['thres001'] = acc_threshold(self.pred_depth, self.gt_depth, valid_mask, threshold=0.01)

        return out_dict


def acc_threshold(depth_pred, depth_gt, mask, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    errors = np.abs((depth_pred[mask] - depth_gt[mask]))
    acc_mask = errors < threshold
    return acc_mask.astype('float').mean()
    