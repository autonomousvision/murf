import numpy as np

import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F


class RandomCrop(object):
    def __init__(self, crop_height, crop_width, fixed_crop=False,
                 with_batch_dim=False,
                 ):
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.with_batch_dim = with_batch_dim

        # center crop, debug purpose
        self.fixed_crop = fixed_crop

    def __call__(self, sample):
        # [V, 3, H, W] or [B, V, 3, H, W]
        size_index = 3 if self.with_batch_dim else 2
        ori_height, ori_width = sample['images'].shape[size_index:]
        assert self.crop_height <= ori_height and self.crop_width <= ori_width

        if self.fixed_crop:
            offset_x = (ori_width - self.crop_width) // 2
            offset_y = (ori_height - self.crop_height) // 2
        else:
            # random crop
            offset_x = np.random.randint(ori_width - self.crop_width + 1)
            offset_y = np.random.randint(ori_height - self.crop_height + 1)

        # crop images
        if self.with_batch_dim:
            sample['images'] = sample['images'][:, :, :, offset_y:offset_y + self.crop_height,
                                                offset_x:offset_x + self.crop_width]
        else:
            sample['images'] = sample['images'][:, :, offset_y:offset_y + self.crop_height,
                                                offset_x:offset_x + self.crop_width]

        # update intrinsics
        if isinstance(sample['intrinsics'], torch.Tensor):
            intrinsics = sample['intrinsics'].clone()  # [V, 3, 3]
        else:
            intrinsics = sample['intrinsics'].copy()  # [V, 3, 3]

        if self.with_batch_dim:
            intrinsics[:, :, 0, 2] = intrinsics[:, :, 0, 2] - offset_x
            intrinsics[:, :, 1, 2] = intrinsics[:, :, 1, 2] - offset_y
        else:
            intrinsics[:, 0, 2] = intrinsics[:, 0, 2] - offset_x
            intrinsics[:, 1, 2] = intrinsics[:, 1, 2] - offset_y

        sample['intrinsics'] = intrinsics

        # update size
        if isinstance(sample['img_wh'], torch.Tensor):
            img_wh = sample['img_wh'].clone()
        else:
            img_wh = sample['img_wh'].copy()

        if self.with_batch_dim:
            img_wh[:, 0] = self.crop_width
            img_wh[:, 1] = self.crop_height
        else:
            img_wh[0] = self.crop_width
            img_wh[1] = self.crop_height

        sample['img_wh'] = img_wh

        # crop depth
        if 'depth' in sample:
            sample['depth'] = sample['depth'][offset_y:offset_y +
                                              self.crop_height, offset_x:offset_x + self.crop_width]

        if 'depth_gt' in sample:
            sample['depth_gt'] = sample['depth_gt'][offset_y:offset_y +
                                                    self.crop_height, offset_x:offset_x + self.crop_width]

        return sample


class RandomResize(object):
    def __init__(self, prob=0.5,
                 crop_height=256,
                 crop_width=384,
                 max_crop_height=None,
                 max_crop_width=None,
                 min_scale=0.8,
                 max_scale=1.2,
                 min_crop_height=None,
                 min_crop_width=None,
                 ):
        self.prob = prob
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_crop_height = max_crop_height
        self.max_crop_width = max_crop_width
        self.min_crop_height = min_crop_height
        self.min_crop_width = min_crop_width

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            ori_height, ori_width = sample['images'].shape[2:]
            # print(ori_height, ori_width)

            if self.max_crop_height is not None and self.max_crop_width is not None:
                # recompute min_scale and max_scale, used for mvimgnet dataset
                if ori_height < ori_width:
                    min_scale = max((self.max_crop_height + 64) /
                                    ori_height, (self.max_crop_width + 64) / ori_width)
                    max_scale = max(min((self.max_crop_height + 128) / ori_height,
                                    (self.max_crop_width + 128) / ori_width), min_scale + 0.01)
                else:
                    min_scale = max((self.max_crop_width + 64) / ori_height,
                                    (self.max_crop_height + 64) / ori_width)
                    max_scale = max(min((self.max_crop_width + 128) / ori_height,
                                    (self.max_crop_height + 128) / ori_width), min_scale + 0.01)
                    # print(min_scale, max_scale)

                scale_factor = np.random.uniform(min_scale, max_scale)
            elif self.min_crop_height is not None and self.min_crop_width is not None:
                # recompute min_scale and max_scale, used for mvimgnet dataset
                if ori_height < ori_width:
                    min_scale = max((self.min_crop_height + 64) /
                                    ori_height, (self.min_crop_width + 64) / ori_width)
                    max_scale = max(min((self.min_crop_height + 128) / ori_height,
                                    (self.min_crop_width + 128) / ori_width), min_scale + 0.05)
                else:
                    min_scale = max((self.min_crop_width + 64) / ori_height,
                                    (self.min_crop_height + 64) / ori_width)
                    max_scale = max(min((self.min_crop_width + 128) / ori_height,
                                    (self.min_crop_height + 128) / ori_width), min_scale + 0.05)

                scale_factor = np.random.uniform(min_scale, max_scale)
            else:
                min_scale_factor = np.maximum(
                    self.crop_height / ori_height, self.crop_width / ori_width)
                scale_factor = np.random.uniform(
                    self.min_scale, self.max_scale)
                scale_factor = np.maximum(min_scale_factor, scale_factor)

            new_height = int(ori_height * scale_factor)
            new_width = int(ori_width * scale_factor)

            sample['images'] = F.interpolate(sample['images'], size=(
                new_height, new_width), mode='bilinear', align_corners=True)

            if 'depth' in sample:
                sample['depth'] = F.interpolate(sample['depth'].unsqueeze(0).unsqueeze(0), size=(
                    new_height, new_width), mode='nearest', align_corners=True).suqeeze(0).squeeze(0)  # [H, W]

            if 'depth_gt' in sample:
                sample['depth_gt'] = F.interpolate(sample['depth_gt'].unsqueeze(0).unsqueeze(0), size=(
                    new_height, new_width), mode='nearest', align_corners=True).suqeeze(0).squeeze(0)  # [H, W]

            # update intrinsics
            intrinsics = sample['intrinsics'].copy()  # [V, 3, 3]
            intrinsics[:, 0, :] * scale_factor
            intrinsics[:, 1, :] * scale_factor

            sample['intrinsics'] = intrinsics

            # update size
            sample['img_wh'] = np.array([new_width, new_height]).astype('int')

            return sample

        else:
            return sample
