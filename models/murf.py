import torch
from easydict import EasyDict as edict
import torch.nn.functional as torch_F
from tqdm import tqdm
from torch import nn

from .gmflow.multiview_gmflow import MultiViewGMFlow
from .rfdecoder.cond_nerf import CondNeRF
from .rfdecoder.cond_nerf_fine import CondNeRFFine

from .gmflow.utils import sample_features_by_grid
from .utils import compute_grid_indices, compute_view_dir_diff
from misc import camera
from misc.utils import generate_window_grid


class MuRF(torch.nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.nerf_setbg_opaque = False
        self.n_src_views = opts.n_src_views

        # encoder
        self.feat_enc = MultiViewGMFlow().to(opts.device)

        # decoder
        self.nerf_dec = CondNeRF(opts).to(opts.device)

        if getattr(opts, 'with_fine_nerf', False):
            self.feat_enc_fine = MultiViewGMFlow().to(opts.device)

            self.nerf_dec_fine = CondNeRFFine(num_samples=getattr(self.opts, 'num_fine_samples', 16),
                                              ).to(opts.device)

    def forward(self, batch, mode=None, render_video=False, render_path_mode='interpolate'):
        self.n_src_views = batch['images'].size(1) - 1  # [B, (V+1), 3, H, W]
        self.img_hw = batch['img_wh'][0].cpu().numpy().tolist()[::-1]

        ref_images = batch.images[:, :self.n_src_views]
        batch_size, _, _, img_h, img_w = ref_images.shape  # [B, V, 3, H, W]

        # extract multi-view image features
        # list of [B, V, C, H, W], multi-scale, resolution from low to high
        ref_feats_list = self.get_img_feat(ref_images)

        # extract target and source poses
        tgt_pose, ref_poses = self.extract_poses(batch)

        if render_video:
            assert mode in ['test', 'val']
            poses_paths = self.get_video_rendering_path(
                tgt_pose, ref_poses, render_path_mode, self.opts.nerf.video_n_frames, batch)
        else:
            poses_paths = [tgt_pose]

        # render images
        for frame_idx, cur_tgt_pose in enumerate(tqdm(poses_paths, desc="rendering video frame...", leave=False) if render_video else poses_paths):
            batch.ray_idx = torch.arange(img_h*img_w, device=self.opts.device)
            # return:
            # rgb: [B, H*W*D, 3]
            # depth: [B, H*W*D]
            # opacity: [B, H*W*D, 1]
            ret = self.render(self.opts, cur_tgt_pose, ray_idx=batch.ray_idx, mode=mode,
                              ref_poses=ref_poses, ref_images=ref_images, ref_feats_list=ref_feats_list,
                              )

            if frame_idx == 0:
                batch.update(edict({k: [] for k in ret.keys()}))
            for k, v in ret.items():
                batch[k].append(v.detach().cpu() if render_video else v)
            if frame_idx == len(poses_paths) - 1:
                for k in ret.keys():
                    batch[k] = torch.cat(batch[k], dim=0)

        return batch

    def extract_poses(self, batch):
        tgt_pose = {}
        tgt_pose['extrinsics'] = batch.extrinsics[:, -1, :3, :]  # B, 3, 4
        tgt_pose['intrinsics'] = batch.intrinsics[:, -1]  # B, 3, 3
        tgt_pose['near_fars'] = batch.near_fars[:, -1]  # B, 2

        ref_poses = {}
        ref_poses['extrinsics'] = batch.extrinsics[:, :-1, :3, :]  # B, N, 3, 4
        ref_poses['intrinsics'] = batch.intrinsics[:, :-1]  # B, N, 3, 3
        ref_poses['near_fars'] = batch.near_fars[:, :-1]  # B, N, 2

        return tgt_pose, ref_poses

    def render(self, opt, tgt_pose, ray_idx=None, mode=None,
               ref_poses=None, ref_images=None, ref_feats_list=None,
               ):
        batch_size, _, _, img_h, img_w = ref_images.shape

        # casting ray with the target camera parameters
        center, ray = camera.get_center_and_ray(img_h, img_w, tgt_pose['extrinsics'], intr=tgt_pose['intrinsics'],
                                                legacy=True, device=opt.device)  # [B,HW,3]

        curr_h, curr_w = None, None

        # original resolution ray and depth
        ray_all = ray.clone()
        center_all = center.clone()
        depth_samples_all = self.sample_depth(opt, batch_size, num_rays=ray.shape[1], near_far=tgt_pose['near_fars'],
                                              legacy=True, mode=mode)  # [B,HW,D,1]

        curr_h, curr_w = self.img_hw

        if self.training and getattr(opt, 'random_crop', False):
            curr_h, curr_w = opt.crop_height, opt.crop_width

        assert ray_idx.shape[0] == curr_h * \
            curr_w, print(ray_idx.shape[0], curr_h, curr_w)

        ray_idx = ray_idx.view(curr_h, curr_w)

        # subsample
        ray_idx = ray_idx[::opt.radiance_subsample_factor,
                          ::opt.radiance_subsample_factor]
        ray_idx = ray_idx.reshape(-1)
        center, ray = center[:, ray_idx], ray[:, ray_idx]

        depth_samples = self.sample_depth(opt, batch_size, num_rays=ray.shape[1], near_far=tgt_pose['near_fars'],
                                          legacy=True, mode=mode,
                                          )  # [B,HW,D,1]
        pts_3D = camera.get_3D_points_from_depth(
            opt, center, ray, depth_samples, multi_samples=True)  # [B,HW,D,3]

        # full image forward
        if not getattr(opt, 'inference_splits', False) or self.training:
            cond_info = self.query_cond_info(
                pts_3D, ref_poses, ref_images, ref_feats_list)

            view_dir_diff = compute_view_dir_diff(
                pts_3D, tgt_pose['extrinsics'], ref_poses['extrinsics'],
            )  # [B, V, H*W, D, 4]
            cond_info['viewdir_diff'] = view_dir_diff

            rgb_samples, density_samples = self.nerf_dec(self.opts,
                                                         cond_info=cond_info,
                                                         img_hw=self.img_hw,
                                                         num_views=self.n_src_views,
                                                         )

            if getattr(opt, 'radiance_subsample_factor', False):
                ray = ray_all
                depth_samples = depth_samples_all
                center = center_all

            # volume rendering to get final 2D output
            rgb, depth, opacity, prob = self.nerf_dec.composite(self.opts, ray, rgb_samples, density_samples, depth_samples,
                                                                setbg_opaque=self.nerf_setbg_opaque)

        # patch-wise inference
        else:
            patch_overlap = getattr(opt, 'patch_overlap', 8)

            num_splits = opt.inference_splits
            b = pts_3D.size(0)
            d = pts_3D.size(2)
            # for loop
            ori_hw = self.img_hw
            curr_h = self.img_hw[0] // opt.radiance_subsample_factor
            curr_w = self.img_hw[1] // opt.radiance_subsample_factor
            upsample_factor = opt.radiance_subsample_factor

            assert curr_h % num_splits == 0 and curr_w % num_splits == 0, \
                'the volume size should be divisable by infernece splits, try to addjust the `--resize_factor` parameters,\
                     or simply set resize_factor = subsample_factor * inference_splits'

            patch_size_h = curr_h // num_splits
            patch_size_w = curr_w // num_splits

            # patch overlap should be smaller than the half of patch size
            patch_overlap = min(patch_size_w // 2,
                                min(patch_size_h // 2, patch_overlap))

            # new patch size includes overlap
            patch_size_h += patch_overlap
            patch_size_w += patch_overlap

            # list of [h, w], patch starting locations
            hws = compute_grid_indices(
                [curr_h, curr_w], [patch_size_h, patch_size_w], min_overlap=patch_overlap)

            assert len(hws) == num_splits ** 2

            sum_rgb = pts_3D.new_zeros((b, *ori_hw, 3))  # [B, H, W, 3]
            sum_depth = pts_3D.new_zeros((b, *ori_hw))  # [B, H, W]
            # [B, H, W, D], prob: [B, H*W, D, 1]
            sum_prob = pts_3D.new_zeros((b, *ori_hw, d))
            sum_opacity = pts_3D.new_zeros((b, *ori_hw))  # [B, H, W]
            weight_rgb = pts_3D.new_zeros((b, *ori_hw))  # [B, H, W]
            weight_depth = pts_3D.new_zeros((b, *ori_hw))  # [B, H, W]
            weight_prob = pts_3D.new_zeros((b, *ori_hw, d))  # [B, H, W]
            weight_opacity = pts_3D.new_zeros((b, *ori_hw))  # [B, H, W]

            # query conditional inputs
            reshape_pts_3D = pts_3D.view(
                b, curr_h, curr_w, d, 3)  # [B, H, W, D, 3]

            # original resolution
            ray = ray_all.view(b, *ori_hw, 3)  # [B, H, W, 3]
            depth_samples = depth_samples_all.view(
                b, *ori_hw, d, 1)  # [B, H, W, D, 1]

            for (h, w) in hws:
                start_h, end_h = h, h + patch_size_h
                start_w, end_w = w, w + patch_size_w

                curr_pts_3D = reshape_pts_3D[:, start_h:end_h, start_w:end_w].reshape(
                    b, -1, d, 3)  # [B, h*w, d, 3]

                curr_cond_info = self.query_cond_info(curr_pts_3D, ref_poses, ref_images, ref_feats_list,
                                                      )

                view_dir_diff = compute_view_dir_diff(
                    curr_pts_3D, tgt_pose['extrinsics'], ref_poses['extrinsics'],
                )  # [B, V, N, D, 4]
                curr_cond_info['viewdir_diff'] = view_dir_diff

                curr_rgb_samples, curr_density_samples = self.nerf_dec(self.opts,
                                                                       cond_info=curr_cond_info,
                                                                       img_hw=[
                                                                           patch_size_h * upsample_factor, patch_size_w * upsample_factor],
                                                                       num_views=self.n_src_views,
                                                                       )  # [B, h*w, D, 3], [B, h*w, D], upsampled

                # volume rendering to get final 2D output
                rgb, depth, opacity, prob = self.nerf_dec.composite(self.opts,
                                                                    ray[:, start_h*upsample_factor:end_h*upsample_factor, start_w *
                                                                        upsample_factor:end_w*upsample_factor].reshape(b, -1, 3),
                                                                    curr_rgb_samples, curr_density_samples,
                                                                    depth_samples[:, start_h*upsample_factor:end_h*upsample_factor,
                                                                                  start_w*upsample_factor:end_w*upsample_factor].reshape(b, -1, d, 1),
                                                                    setbg_opaque=self.nerf_setbg_opaque)

                sum_rgb[:, start_h*upsample_factor:end_h*upsample_factor, start_w*upsample_factor:end_w *
                        upsample_factor] += rgb.view(b, patch_size_h * upsample_factor, patch_size_w * upsample_factor, 3)
                sum_depth[:, start_h*upsample_factor:end_h*upsample_factor, start_w*upsample_factor:end_w *
                          upsample_factor] += depth.view(b, patch_size_h * upsample_factor, patch_size_w * upsample_factor)
                sum_prob[:, start_h*upsample_factor:end_h*upsample_factor, start_w*upsample_factor:end_w *
                         upsample_factor] += prob.view(b, patch_size_h * upsample_factor, patch_size_w * upsample_factor, d)
                sum_opacity[:, start_h*upsample_factor:end_h*upsample_factor, start_w*upsample_factor:end_w *
                            upsample_factor] += opacity.view(b, patch_size_h * upsample_factor, patch_size_w * upsample_factor)

                weight_rgb[:, start_h*upsample_factor:end_h*upsample_factor,
                           start_w*upsample_factor:end_w*upsample_factor] += 1
                weight_depth[:, start_h*upsample_factor:end_h*upsample_factor,
                             start_w*upsample_factor:end_w*upsample_factor] += 1
                weight_prob[:, start_h*upsample_factor:end_h*upsample_factor,
                            start_w*upsample_factor:end_w*upsample_factor] += 1
                weight_opacity[:, start_h*upsample_factor:end_h*upsample_factor,
                               start_w*upsample_factor:end_w*upsample_factor] += 1

            # merge with simple average
            rgb = (sum_rgb / weight_rgb.unsqueeze(-1)
                   ).view(b, ori_hw[0]*ori_hw[1], 3)
            depth = (sum_depth / weight_depth).view(b, ori_hw[0]*ori_hw[1])
            prob = (sum_prob / weight_prob).view(b, ori_hw[0]*ori_hw[1], d, 1)
            opacity = (sum_opacity / weight_opacity).view(b,
                                                          ori_hw[0]*ori_hw[1], 1)

            # used for fine nerf later
            ray = ray.view(b, -1, 3)  # [B, H*W, 3]
            depth_samples = depth_samples.view(b, -1, d, 1)  # [B, H*W, D, 1]
            center = center_all

        ret = edict(rgb=rgb, depth=depth, opacity=opacity)  # [B,HW,K]

        # coarse to fine nerf
        if getattr(opt, 'with_fine_nerf', False):
            # resample depth from the coarse results
            # depth_samples: [B, H*W, N, 1]
            # prob: [B, H*W, N, 1]
            z_vals_mid = .5 * \
                (depth_samples[..., -1][..., 1:] +
                 depth_samples[..., -1][..., :-1])  # [B, H*W, N-1]
            weights = prob[..., -1][..., 1:-1]
            num_fine_samples = getattr(opt, 'num_fine_samples', 16)

            # remove depth boundary since they can't provide too much useful information
            pad = getattr(opt, 'fine_sample_pad', 2)
            num_fine_samples = num_fine_samples + pad

            depth_samples = self.sample_pdf(
                z_vals_mid, weights, N_samples=num_fine_samples, det=True).unsqueeze(-1)  # [B, H*W, N, 1]

            # clamp depth
            depth_min, depth_max = torch.split(
                tgt_pose['near_fars'], [1, 1], dim=-1)
            depth_samples = depth_samples.clamp(
                depth_min[:, None, None], depth_max[:, None, None])

            # remove depth boundary since they can't provide too much useful information
            depth_samples = depth_samples[:, :, (pad // 2):-(pad // 2)]

            pts_3D = camera.get_3D_points_from_depth(
                opt, center, ray, depth_samples, multi_samples=True)  # [B,HW,N,3]

            # include coarse color prediction as another source view
            # construct batch images
            fine_ref_poses = {}
            coarse_color_pred = rgb.detach().view(pts_3D.size(
                0), self.img_hw[0], self.img_hw[1], 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
            for k, v in ref_poses.items():
                fine_ref_poses[k] = torch.cat(
                    (v, tgt_pose[k].unsqueeze(1)), dim=1)  # [B, (V+1), ...]

            # [B, (V+1), 3, H, W]
            ref_images = torch.cat(
                (ref_images, coarse_color_pred.unsqueeze(1)), dim=1)

            ref_feats_list = self.get_img_feat(
                ref_images,
                is_fine_nerf=True,
            )

            # project features to lower dim
            assert hasattr(self.nerf_dec_fine, 'feature_proj')
            assert len(ref_feats_list) == len(self.nerf_dec_fine.feature_proj)
            # list of [B, V, C,, H, W], resolution low to high
            for i in range(len(ref_feats_list)):
                b_, v_, c_, h_, w_ = ref_feats_list[i].shape
                ref_feats_list[i] = self.nerf_dec_fine.feature_proj[i](
                    ref_feats_list[i].view(b_ * v_, c_, h_, w_)).view(b_, v_, -1, h_, w_)

            if not getattr(opt, 'fine_inference_splits', False) or self.training:
                cond_info_fine = self.query_cond_info(pts_3D, fine_ref_poses, ref_images, ref_feats_list,
                                                      is_fine_nerf=getattr(
                                                          opt, 'with_fine_nerf', False),
                                                      )

                view_dir_diff = compute_view_dir_diff(
                    pts_3D, tgt_pose['extrinsics'], fine_ref_poses['extrinsics'],
                )  # [B, V, N, D, 4]
                cond_info_fine['viewdir_diff'] = view_dir_diff

                rgb_samples, density_samples = self.nerf_dec_fine(cond_info=cond_info_fine,
                                                                  img_hw=self.img_hw,
                                                                  )

                # original resolution depth samples
                ray = ray_all
                h, w = self.img_hw

                rgb, depth, opacity, prob = self.nerf_dec_fine.composite(self.opts, ray, rgb_samples, density_samples, depth_samples,
                                                                         setbg_opaque=self.nerf_setbg_opaque,
                                                                         render_depth_no_boundary=2,
                                                                         )

            # patch-wise inference
            else:
                patch_overlap = getattr(opt, 'fine_patch_overlap', 8)

                num_splits = opt.fine_inference_splits
                b = pts_3D.size(0)
                d = pts_3D.size(2)
                # for loop
                ori_hw = self.img_hw
                curr_h = self.img_hw[0]
                curr_w = self.img_hw[1]

                assert curr_h % num_splits == 0 and curr_w % num_splits == 0, \
                    'fine nerf: the volume size should be divisable by infernece splits, try to addjust the `--resize_factor` parameters, or simply set resize_factor = inference_splits'

                patch_size_h = curr_h // num_splits
                patch_size_w = curr_w // num_splits

                # patch overlap should be smaller than half patch size
                patch_overlap = min(patch_size_w // 2,
                                    min(patch_size_h // 2, patch_overlap))

                # new patch size includes overlap such that not too many patches
                patch_size_h += patch_overlap
                patch_size_w += patch_overlap

                # list of [h, w], patch starting locations
                hws = compute_grid_indices(
                    [curr_h, curr_w], [patch_size_h, patch_size_w], min_overlap=patch_overlap)

                assert len(hws) == num_splits ** 2

                sum_rgb = pts_3D.new_zeros((b, *ori_hw, 3))  # [B, H, W, 3]
                sum_depth = pts_3D.new_zeros((b, *ori_hw))  # [B, H, W]
                # [B, H, W, D], prob: [B, H*W, D, 1]
                sum_prob = pts_3D.new_zeros((b, *ori_hw, d))
                sum_opacity = pts_3D.new_zeros((b, *ori_hw))  # [B, H, W]
                weight_rgb = pts_3D.new_zeros((b, *ori_hw))  # [B, H, W]
                weight_depth = pts_3D.new_zeros((b, *ori_hw))  # [B, H, W]
                weight_prob = pts_3D.new_zeros((b, *ori_hw, d))  # [B, H, W]
                weight_opacity = pts_3D.new_zeros((b, *ori_hw))  # [B, H, W]

                # query conditional inputs
                reshape_pts_3D = pts_3D.view(
                    b, curr_h, curr_w, d, 3)  # [B, H, W, D, 3]

                # original resolution
                ray = ray_all.view(b, *ori_hw, 3)  # [B, H, W, 3]
                depth_samples = depth_samples.view(
                    b, *ori_hw, d, 1)  # [B, H, W, D, 1]

                for (h, w) in hws:
                    start_h, end_h = h, h + patch_size_h
                    start_w, end_w = w, w + patch_size_w

                    curr_pts_3D = reshape_pts_3D[:, start_h:end_h, start_w:end_w].reshape(
                        b, -1, d, 3)  # [B, h*w, d, 3]

                    cond_info_fine = self.query_cond_info(curr_pts_3D, fine_ref_poses, ref_images, ref_feats_list,
                                                          is_fine_nerf=getattr(
                                                              opt, 'with_fine_nerf', False),
                                                          )

                    view_dir_diff = compute_view_dir_diff(
                        curr_pts_3D, tgt_pose['extrinsics'], fine_ref_poses['extrinsics'],
                    )  # [B, V, N, D, 4]
                    cond_info_fine['viewdir_diff'] = view_dir_diff

                    curr_rgb_samples, curr_density_samples = self.nerf_dec_fine(cond_info=cond_info_fine,
                                                                                img_hw=(
                                                                                    patch_size_h, patch_size_w),
                                                                                )

                    # original resolution
                    upsample_factor = 1

                    rgb, depth, opacity, prob = self.nerf_dec_fine.composite(self.opts,
                                                                             ray[:, start_h*upsample_factor:end_h*upsample_factor, start_w *
                                                                                 upsample_factor:end_w*upsample_factor].reshape(b, -1, 3),
                                                                             curr_rgb_samples, curr_density_samples,
                                                                             depth_samples[:, start_h*upsample_factor:end_h*upsample_factor,
                                                                                           start_w*upsample_factor:end_w*upsample_factor].reshape(b, -1, d, 1),
                                                                             setbg_opaque=self.nerf_setbg_opaque,
                                                                             render_depth_no_boundary=2,
                                                                             )

                    sum_rgb[:, start_h*upsample_factor:end_h*upsample_factor, start_w*upsample_factor:end_w *
                            upsample_factor] += rgb.view(b, patch_size_h * upsample_factor, patch_size_w * upsample_factor, 3)
                    sum_depth[:, start_h*upsample_factor:end_h*upsample_factor, start_w*upsample_factor:end_w *
                              upsample_factor] += depth.view(b, patch_size_h * upsample_factor, patch_size_w * upsample_factor)
                    sum_prob[:, start_h*upsample_factor:end_h*upsample_factor, start_w*upsample_factor:end_w *
                             upsample_factor] += prob.view(b, patch_size_h * upsample_factor, patch_size_w * upsample_factor, d)
                    sum_opacity[:, start_h*upsample_factor:end_h*upsample_factor, start_w*upsample_factor:end_w *
                                upsample_factor] += opacity.view(b, patch_size_h * upsample_factor, patch_size_w * upsample_factor)

                    weight_rgb[:, start_h*upsample_factor:end_h*upsample_factor,
                               start_w*upsample_factor:end_w*upsample_factor] += 1
                    weight_depth[:, start_h*upsample_factor:end_h*upsample_factor,
                                 start_w*upsample_factor:end_w*upsample_factor] += 1
                    weight_prob[:, start_h*upsample_factor:end_h*upsample_factor,
                                start_w*upsample_factor:end_w*upsample_factor] += 1
                    weight_opacity[:, start_h*upsample_factor:end_h*upsample_factor,
                                   start_w*upsample_factor:end_w*upsample_factor] += 1

                # merge with simple average
                rgb = (sum_rgb / weight_rgb.unsqueeze(-1)
                       ).view(b, ori_hw[0]*ori_hw[1], 3)
                depth = (sum_depth / weight_depth).view(b, ori_hw[0]*ori_hw[1])
                opacity = (sum_opacity / weight_opacity).view(b,
                                                              ori_hw[0]*ori_hw[1], 1)

            ret['rgb'] = rgb
            ret['depth'] = depth
            ret['opacity'] = opacity

        return ret

    def sample_depth(self, opt, batch_size, num_rays, near_far, legacy=False, mode='train',
                     depth_subsample_factor=None,
                     num_samples=None,
                     ):
        depth_min, depth_max = torch.split(near_far, [1, 1], dim=-1)

        if getattr(opt, 'log_sampler', False):
            depth_min, depth_max = torch.log(depth_min), torch.log(depth_max)

        rand_shift = 0. if legacy else 0.5

        ray_samples = opt.nerf.sample_intvs if num_samples is None else num_samples

        if depth_subsample_factor is not None:
            assert ray_samples % depth_subsample_factor == 0
            ray_samples = ray_samples // depth_subsample_factor

        depth_denom = ray_samples - 1 if legacy else ray_samples

        if mode == 'train' and opt.nerf.sample_stratified:
            rand_samples = torch.rand(
                batch_size, num_rays, ray_samples, 1, device=opt.device)  # [B, HW, N, 1]
        else:
            rand_samples = rand_shift * \
                torch.ones(batch_size, num_rays,
                           ray_samples, 1, device=opt.device)

        rand_samples = rand_samples + \
            torch.arange(ray_samples, device=opt.device)[
                None, None, :, None].float()  # [B,HW,N,1]
        depth_max = depth_max.reshape(
            batch_size, *[1]*(rand_samples.dim() - 1))  # [B, 1, 1, 1]
        depth_min = depth_min.reshape(
            batch_size, *[1]*(rand_samples.dim() - 1))
        depth_samples = rand_samples / depth_denom * \
            (depth_max - depth_min) + depth_min  # [B,HW,N,1]  # for +0.5

        if getattr(opt, 'log_sampler', False):
            depth_samples = torch.exp(depth_samples)

        depth_samples = dict(
            metric=depth_samples,
            inverse=1/(depth_samples+1e-8),
        )[opt.nerf.depth.param]

        return depth_samples

    def sample_pdf(self, bins, weights, N_samples, det=False):
        weights = weights + 1e-5
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        # [B, H*W, N]
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

        if det:
            u = torch.linspace(0., 1., steps=N_samples).to(cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(cdf.device)

        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (B, H*W, N, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1],
                         inds_g.shape[2], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(2).expand(matched_shape), 3, inds_g)
        bins_g = torch.gather(bins.unsqueeze(
            2).expand(matched_shape), 3, inds_g)

        denom = (cdf_g[..., 1]-cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[..., 0])/denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

        samples = samples.detach()

        return samples

    def get_img_feat(self, imgs, is_fine_nerf=False):
        if is_fine_nerf:
            assert hasattr(self, 'feat_enc_fine')
            feat_enc = self.feat_enc_fine
        else:
            feat_enc = self.feat_enc

        out_dict = feat_enc(
            images=imgs, attn_splits_list=self.opts.encoder.attn_splits_list)
        img_feat_list = out_dict['aug_feats_list']

        return img_feat_list

    def query_cond_info(self, point_samples, ref_poses, ref_images, ref_feats_list,
                        is_fine_nerf=False,
                        ):
        batch_size, n_views, _, img_h, img_w = ref_images.shape

        device = self.opts.device
        cos_n_group = [
            1, 1, 1, 1] if is_fine_nerf else self.opts.encoder.cos_n_group

        cos_n_group = [cos_n_group] if isinstance(
            cos_n_group, int) else cos_n_group
        feat_data_list = [[] for _ in range(len(ref_feats_list))]
        color_data = []

        # query information from each source view
        inv_scale = torch.tensor(
            [[img_w - 1, img_h - 1]]).repeat(batch_size, 1).to(device)

        for view_idx in range(n_views):
            near_far_ref = ref_poses['near_fars'][:, view_idx]
            extr_ref, intr_ref = ref_poses['extrinsics'][:, view_idx].clone(
            ), ref_poses['intrinsics'][:, view_idx].clone()
            point_samples_pixel = camera.get_coord_ref_ndc(extr_ref, intr_ref, point_samples,
                                                           inv_scale, near_far=near_far_ref)
            grid = point_samples_pixel[..., :2] * 2.0 - 1.0  # [B, H*W, D, 2]

            # query features from each view
            # ref_feats_list: list of [B, V, C, H, W], resolution from low to high
            for scale_idx, img_feat_cur_scale in enumerate(ref_feats_list):
                raw_whole_feats = img_feat_cur_scale[:, view_idx]

                sampled_feats = sample_features_by_grid(
                    raw_whole_feats, grid, align_corners=True, mode='bilinear', padding_mode='border')
                feat_data_list[scale_idx].append(sampled_feats)

            # sample window for color
            if getattr(self.opts, 'sample_color_window_radius', False) and not is_fine_nerf:
                local_radius = self.opts.sample_color_window_radius
                assert local_radius > 0
                local_h = 2 * local_radius + 1
                local_w = 2 * local_radius + 1

                window_grid = generate_window_grid(-local_radius, local_radius,
                                                   -local_radius, local_radius,
                                                   local_h, local_w, device=grid.device)  # [2R+1, 2R+1, 2]

                b, n, d = grid.shape[:3]
                # grid is in [-1, 1]
                color_sample_grid = grid.view(
                    b, n * d, 1, 2)  # [B, H*W*D, 1, 2]
                color_sample_grid = (color_sample_grid + 1) / 2  # [0, 1]
                color_sample_grid = torch.cat((color_sample_grid[:, :, :, :1] * (
                    img_w - 1), color_sample_grid[:, :, :, 1:] * (img_h - 1)), dim=-1)  # image scale

                # [B, 1, (2R+1)^2, 2]
                window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)
                color_sample_grid = color_sample_grid + \
                    window_grid  # [B, H*W*D, (2R+1)^2, 2]

                # normalize to [-1, 1]
                c = torch.Tensor([(img_w - 1) / 2., (img_h - 1) / 2.]
                                 ).float().to(color_sample_grid.device)
                color_sample_grid = (color_sample_grid - c) / c  # [-1, 1]

                sampled_color = torch_F.grid_sample(
                    ref_images[:, view_idx], color_sample_grid, align_corners=True, mode='bilinear', padding_mode='border')  # [B, 3, H*W*D, (2R+1)^2]

                sampled_color = sampled_color.permute(0, 1, 3, 2).reshape(
                    b, -1, n, d)  # [B, (2R+1)^2 * 3, H*W, D]

                color_data.append(sampled_color)
            else:
                # query color
                color_data.append(torch_F.grid_sample(
                    ref_images[:, view_idx], grid, align_corners=True, mode='bilinear', padding_mode='border'))

        # merge queried information from all views
        all_data = {}

        # concat all features
        # list of [B, V, C, H*W, D], different scales
        all_features = [torch.stack(x, dim=1) for x in feat_data_list]
        # [B, V, C', H*W, D], concat all scale features in channel
        sampled_feature_data = torch.cat(all_features, dim=2)

        # [B, V, C, HW, D]
        all_data['sampled_feature_info'] = sampled_feature_data

        # compute cosine similarities
        merged_feat_data = []
        for feat_data_idx, raw_feat_data in enumerate(feat_data_list):
            n_feat_data = torch.stack(
                raw_feat_data, dim=1)  # [B, V, C, H*W, D]
            iB, iN, iC, iR, iP = n_feat_data.shape
            n_feat_data = n_feat_data.reshape(
                iB, iN, cos_n_group[feat_data_idx], iC // cos_n_group[feat_data_idx], iR, iP)
            n_feat_data = n_feat_data.permute(
                0, 2, 4, 5, 1, 3).reshape(-1, iN, iC // cos_n_group[feat_data_idx])
            norm_n_feat = torch.nn.functional.normalize(n_feat_data, dim=-1)
            norm_n_feat_t = norm_n_feat.permute(0, 2, 1)
            cos_sims_all = torch.matmul(norm_n_feat, norm_n_feat_t)  # BxNxN
            triu_i = torch.triu_indices(iN, iN, 1)
            # index value in the upper triagnular
            cos_sims = cos_sims_all[:, triu_i[0], triu_i[1]]
            # [B, n_pairs, n_groups, H*W, D]
            cur_updated_feat_data = cos_sims.reshape(
                iB, cos_n_group[feat_data_idx], iR, iP, -1).permute(0, 4, 1, 2, 3)

            if getattr(self.opts, 'weighted_cosine', False) and not is_fine_nerf:
                pass
            else:
                cur_updated_feat_data = torch.mean(
                    cur_updated_feat_data, dim=1)  # [B, G, H*W, D]

            merged_feat_data.append(cur_updated_feat_data)

        if getattr(self.opts, 'weighted_cosine', False) and not is_fine_nerf:
            merged_feat_data = torch.cat(
                merged_feat_data, dim=2)  # [B, P, C, H*W, D]
        else:
            merged_feat_data = torch.cat(merged_feat_data, dim=1)

        all_data['feat_info'] = merged_feat_data

        # merge sampled color
        # [B, (2R+1)^2 * 3, H*W, D, V]
        merged_color_data = torch.stack(color_data, dim=-1)
        all_data['color_info'] = merged_color_data

        for k, v in all_data.items():
            if k != 'sampled_feature_info':
                if k == 'color_info':
                    # [B, H*W, D, (2R+1)^2 * 3, V]
                    all_data[k] = v.permute(0, 2, 3, 1, 4)
                elif getattr(self.opts, 'weighted_cosine', False) and k == 'feat_info' and not is_fine_nerf:
                    all_data[k] = all_data[k].permute(
                        0, 1, 4, 3, 2)  # [B, V, D, H*W, C]
                else:
                    all_data[k] = v.permute(0, 2, 3, 1)

        return all_data

    def get_video_rendering_path(self, tgt_pose, ref_poses, mode, n_frames=30, batch=None):
        poses_paths = []
        for batch_idx, cur_src_poses in enumerate(ref_poses['extrinsics']):
            if mode == 'interpolate':
                # convert to c2ws
                # print(cur_src_poses.shape)  # [V, 3, 4]
                pose_square = torch.eye(4).unsqueeze(0).repeat(
                    cur_src_poses.shape[0], 1, 1).to(self.opts.device)
                pose_square[:, :3, :] = cur_src_poses
                cur_c2ws = pose_square.double().inverse()[:, :3, :].to(
                    torch.float32).cpu().detach().numpy()
                cur_path = camera.get_interpolate_render_path(
                    cur_c2ws, n_frames)  # [N, 4, 4]
            elif mode == 'spiral':
                assert batch is not None, "Must provide all c2ws and near_far for getting spiral rendering path."
                cur_c2ws_all = batch['c2ws_all'][batch_idx].detach(
                ).cpu().numpy()
                cur_near_far = tgt_pose['near_fars'][batch_idx].detach(
                ).cpu().numpy().tolist()
                rads_scale = getattr(
                    self.opts.nerf, "video_rads_scale", 0.1)
                cur_path = camera.get_spiral_render_path(
                    cur_c2ws_all, cur_near_far, rads_scale=rads_scale, N_views=n_frames)
            else:
                raise Exception(
                    f'Unknown video rendering path mode {mode}')

            cur_w2cs = torch.tensor(cur_path).inverse()[:, :3].to(
                torch.float32).to(self.opts.device)
            poses_paths.append(cur_w2cs)

        poses_paths = torch.stack(poses_paths, dim=0)  # [B, N, 3, 4]
        intrinsics = tgt_pose['intrinsics']

        n_frames = poses_paths.shape[1]

        updated_tgt_poses = []
        for frame_idx in range(n_frames):
            updated_tgt_poses.append(dict(extrinsics=poses_paths[:, frame_idx],
                                          intrinsics=intrinsics.clone().detach(),
                                          near_fars=tgt_pose['near_fars'].clone().detach()))

        return updated_tgt_poses



