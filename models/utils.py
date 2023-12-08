import numpy as np
import math
import torch


def compute_grid_indices(image_shape, patch_size, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_view_dir_diff(xyz, tgt_extrinsics, src_extrinsics):
    # xyz: [B, N, D, 3]
    # tgt_extrinsics: [B, 3, 4]
    # src_extrinsics: [B, V, 3, 4]

    b, n, d = xyz.shape[:3]
    num_views = src_extrinsics.size(1)
    tgt_camera_pos = - \
        torch.bmm(tgt_extrinsics[:, :, :3].inverse(),
                  tgt_extrinsics[:, :, 3:])[:, :, -1]  # [B, 3]
    src_camera_pos = [-torch.bmm(src_extrinsics[:, i, :, :3].inverse(), src_extrinsics[:, i, :, 3:])[
        :, :, -1] for i in range(num_views)]  # list of [B, 3]
    tgt_diff = xyz - tgt_camera_pos[:, None, None]  # [B, N, D, 3]
    src_diff = [xyz - src_camera_pos[i][:, None, None]
                for i in range(num_views)]  # list of [B, N, D, 3]

    tgt_diff = tgt_diff / (torch.norm(tgt_diff, dim=-1, keepdim=True) + 1e-6)
    src_diff = [src_diff[i] / (torch.norm(src_diff[i], dim=-1, keepdim=True))
                for i in range(num_views)]

    ray_diff_dot = [torch.sum(tgt_diff * src_diff[i],
                              dim=-1, keepdim=True) for i in range(num_views)]

    ray_diff_dot = torch.stack(ray_diff_dot, dim=1)  # [B, V, N, D, 1]

    ray_diff = [tgt_diff - src_diff[i] for i in range(num_views)]
    ray_diff_norm = [torch.norm(ray_diff[i], dim=-1, keepdim=True)
                     for i in range(num_views)]  # list of [B, N, D, 1]
    ray_diff_dir = [ray_diff[i] / (ray_diff_norm[i] + 1e-6)
                    for i in range(num_views)]

    ray_diff_dir = torch.stack(ray_diff_dir, dim=1)  # [B, V, N, D, 3]

    ray_diff = torch.cat((ray_diff_dir, ray_diff_dot),
                         dim=-1)  # [B, V, N, D, 4]

    return ray_diff

