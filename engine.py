import torch
from easydict import EasyDict as edict
import os
from torch.utils.data import DataLoader
import time
import tqdm
import imageio
from collections import OrderedDict
import numpy as np
import re
import math
import torch.nn.functional as F

from misc.utils import log, compute_image_diff, compute_depth_diff
from misc.metrics import EvalTools
from misc.train_helpers import summarize_loss, summarize_metrics_list
from datasets import datas_dict
from models.murf import MuRF
from misc import utils
from loss import SSIM
import lpips
from misc.depth_viz import viz_depth_tensor
from datasets.ibrnet_mix.create_training_dataset import create_training_dataset
from datasets.transforms import RandomCrop


class Engine():
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.n_src_views = opts.n_src_views
        self.epoch_start = 0
        self.iter_start = 0
        os.makedirs(opts.output_path, exist_ok=True)

    def L1_loss(self, pred, label=0):
        loss = (pred.contiguous() - label).abs()
        return loss.mean()

    def load_dataset(self, splits):
        # load training data
        log.info(f"loading datasets...")
        for split in splits:
            if getattr(self.opts, f'data_{split}', None):
                if split == 'test':
                    data_opts_list = [
                        v for _, v in self.opts.data_test.items()]
                    self.test_loaders = []
                else:
                    data_opts_list = [getattr(self.opts, f'data_{split}')]

                for data_opts in data_opts_list:
                    if data_opts is None:
                        continue
                    scene_list = getattr(data_opts, "scene_list", None)
                    test_views_method = getattr(
                        data_opts, "test_views_method", "nearest")

                    if split == 'train' and getattr(self.opts, 'mix_data_train', False):
                        assert self.opts.dist is True
                        cur_dataset, train_sampler = create_training_dataset(root_dir=data_opts.root_dir,
                                                                             n_views=self.n_src_views,
                                                                             distributed=self.opts.dist,
                                                                             train_dataset=getattr(
                                                                                 self.opts, 'mix_datasets', "llff+spaces+ibrnet_collected+realestate+google_scanned"),
                                                                             dataset_weights=getattr(self.opts, 'dataset_weights', [
                                                                                                     0.3, 0.15, 0.35, 0.15, 0.05]),
                                                                             num_replicas=torch.cuda.device_count(),
                                                                             rank=self.opts.local_rank,
                                                                             mixall=getattr(
                                                                                 self.opts, 'mixall', False),
                                                                             no_random_view=getattr(
                                                                                 self.opts, 'no_random_view', False),
                                                                             dataset_replicas=getattr(
                                                                                 self.opts, 'dataset_replicas', None),
                                                                             realestate_full_set=getattr(
                                                                                 self.opts, 'realestate_full_set', False),
                                                                             realestate_frame_dir=getattr(
                                                                                 self.opts, 'realestate_frame_dir', None),
                                                                             mixall_random_view=getattr(
                                                                                 self.opts, 'mixall_random_view', False),
                                                                             realestate_use_all_scenes=getattr(
                                                                                 self.opts, 'realestate_use_all_scenes', False),
                                                                             )
                        self.train_sampler = train_sampler
                        # only support batch size 1 currently due to different image resolutions in the mixed datasets
                        assert self.opts.batch_size == 1

                    else:
                        cur_dataset = datas_dict[data_opts.dataset_name](data_opts.root_dir,
                                                                         split=split,
                                                                         pose_root=getattr(
                                                                             data_opts, 'pose_dir', None),
                                                                         n_views=self.n_src_views, img_wh=data_opts.img_wh, max_len=data_opts.max_len,
                                                                         scene_list=scene_list,
                                                                         test_views_method=test_views_method,
                                                                         random_crop=(split == 'train' and getattr(
                                                                             self.opts, 'random_crop', False)),
                                                                         crop_height=getattr(
                                                                             self.opts, 'crop_height', None),
                                                                         crop_width=getattr(
                                                                             self.opts, 'crop_width', None),
                                                                         max_crop_height=getattr(
                                                                             self.opts, 'max_crop_height', None),
                                                                         max_crop_width=getattr(
                                                                             self.opts, 'max_crop_width', None),
                                                                         min_crop_height=getattr(
                                                                             self.opts, 'min_crop_height', None),
                                                                         min_crop_width=getattr(
                                                                             self.opts, 'min_crop_width', None),
                                                                         min_views=getattr(
                                                                             self.opts, 'min_views', None),
                                                                         max_views=getattr(
                                                                             self.opts, 'max_views', None),
                                                                         render_scan_id=getattr(
                                                                             data_opts, 'render_scan_id', None),
                                                                         random_resize=(split == 'train' and getattr(
                                                                             self.opts, 'random_resize', False)),
                                                                         test_scan_name=getattr(
                                                                             self.opts, 'test_scan_name', None),
                                                                         fixed_test_set=getattr(
                                                                             self.opts, 'fixed_realestate_test_set', False),
                                                                         frame_distance=getattr(
                                                                             self.opts, 'realestate_frame_distance', 128),
                                                                         view_selection_stride=getattr(
                                                                             self.opts, 'view_selection_stride', 1),
                                                                         img_scale_factor=getattr(
                                                                             self.opts, 'llff_img_scale_factor', 4),
                                                                         test_view_stride=getattr(
                                                                             self.opts, 'dtu_test_view_stride', None),
                                                                         window_size=getattr(
                                                                             self.opts, 'realestate_window_size', 128),
                                                                         continuous_view=getattr(
                                                                             self.opts, 'dtu_continuous_view', False),
                                                                         test_scene_name=getattr(
                                                                             self.opts, 'test_scene_name', None),
                                                                         downsample_factor=getattr(
                                                                             self.opts, 'mipnerf360_downsample_factor', 8),
                                                                         fixed_target_frame=getattr(
                                                                             self.opts, 'realestate_fixed_target_frame', False),
                                                                         )

                        if split == 'train' and self.opts.dist:
                            train_sampler = torch.utils.data.distributed.DistributedSampler(
                                cur_dataset,
                                num_replicas=torch.cuda.device_count(),
                                rank=self.opts.local_rank)
                            self.train_sampler = train_sampler
                        else:
                            train_sampler = None

                    if split == 'train':
                        shuffle = False if self.opts.dist else True
                    else:
                        shuffle = False

                    cur_loader = DataLoader(cur_dataset, shuffle=shuffle, num_workers=data_opts.num_workers,
                                            batch_size=self.opts.batch_size if split == 'train' else 1, pin_memory=True,
                                            sampler=train_sampler,
                                            )

                    if split == 'test':
                        self.test_loaders.append(cur_loader)
                    else:
                        setattr(self, f"{split}_loader", cur_loader)
                    log.info(
                        f"  * loaded {split} set of {data_opts.dataset_name}")

    def build_networks(self):
        log.info("building networks...")
        self.model = MuRF(
            self.opts).to(self.opts.device)
        log.info(self.model)
        if self.opts.encoder.pretrain_weight and (not self.opts.load) and (not self.opts.resume):
            utils.load_gmflow_checkpoint(self.model.feat_enc, self.opts.encoder.pretrain_weight, self.opts.device,
                                         gmflow_n_blocks=self.opts.encoder.num_transformer_layers,
                                         no_strict_load=getattr(
                                             self.opts, 'no_strict_load', False),
                                         )
            log.info(
                f"loaded gmflow pretrained weight for encoder from {self.opts.encoder.pretrain_weight}.")

        if self.opts.dist:
            self.model_ddp = torch.nn.parallel.DistributedDataParallel(
                self.model.to(self.opts.device),
                device_ids=[self.opts.local_rank],
                output_device=self.opts.local_rank,
                find_unused_parameters=False,
            )
            self.model = self.model_ddp.module

        # number of parameters
        num_params = sum([p.numel()
                         for p in self.model.parameters() if p.requires_grad])
        print('Nubmer of parameters: %d' % num_params)

        # load lpips network for loss
        if hasattr(self.opts, 'loss_weight') and getattr(self.opts.loss_weight, 'lpips', False):
            self.lpips_loss_func = lpips.LPIPS(net='vgg').to(self.opts.device)
            for param in self.lpips_loss_func.parameters():
                param.requires_grad = False

        # ssim loss
        if hasattr(self.opts, 'loss_weight') and getattr(self.opts.loss_weight, 'ssim', False):
            self.ssim_loss_func = SSIM(patch_size=getattr(
                self.opts, 'ssim_patch_size', 3)).to(self.opts.device)

    def setup_optimizer(self):
        log.info("setting up optimizers...")
        # load trainable params

        optim_params = [dict(params=self.model.feat_enc.parameters(), lr=self.opts.optim.lr_enc),
                        dict(params=self.model.nerf_dec.parameters(), lr=self.opts.optim.lr_dec)]
        lr_lists = [self.opts.optim.lr_enc, self.opts.optim.lr_dec]

        # fine nerf
        if hasattr(self.model, 'nerf_dec_fine'):
            if getattr(self.opts, 'freeze_coarse_nerf', False):
                for params in self.model.feat_enc.parameters():
                    params.requires_grad = False

                for params in self.model.nerf_dec.parameters():
                    params.requires_grad = False

                optim_params = [
                    dict(params=self.model.nerf_dec_fine.parameters(), lr=self.opts.optim.lr_dec)]
                lr_lists = [self.opts.optim.lr_dec]

                trainable_params = sum(
                    [p.numel() for p in self.model.nerf_dec_fine.parameters() if p.requires_grad])

                # fine encoder
                optim_params.append(
                    dict(params=self.model.feat_enc_fine.parameters(),
                            lr=self.opts.optim.lr_enc)
                )
                lr_lists.append(self.opts.optim.lr_enc)

                trainable_params += sum(
                    [p.numel() for p in self.model.feat_enc_fine.parameters() if p.requires_grad])

                print('Trainable parameters: %d' % trainable_params)
            else:
                optim_params.append(
                    dict(params=self.model.nerf_dec_fine.parameters(), lr=self.opts.optim.lr_dec))
                lr_lists.append(self.opts.optim.lr_dec)

        # set up optimizer
        optim_type = self.opts.optim.algo.type
        optim_kwargs = {k: v for k,
                        v in self.opts.optim.algo.items() if k != "type"}

        self.optim = getattr(torch.optim, optim_type)(
            optim_params, **optim_kwargs)
        info = f"  * {optim_type} optimizer (" + ', '.join(
            [f'{k}={v}' for k, v in optim_kwargs.items()]) + ')'
        log.info(info)

        # set up scheduler if needed
        self.sched_type = None
        if self.opts.optim.sched:
            sched_type = self.opts.optim.sched.type
            sched_kwargs = {k: v for k,
                            v in self.opts.optim.sched.items() if k != "type"}
            info = f"  * {sched_type} scheduler"
            if sched_type == 'OneCycleLR':  # set additional param accordingly
                assert hasattr(
                    self, 'train_loader'), "Must initialize the training data, to calculate total steps for OneCycleLR"

                steps_per_epoch = len(self.train_loader)

                if self.opts.batch_size > 1:
                    sched_kwargs.update(dict(
                        total_steps=self.opts.max_epoch * steps_per_epoch + 50,
                        max_lr=lr_lists,
                    ))
                else:
                    sched_kwargs.update(dict(
                        epochs=self.opts.max_epoch, steps_per_epoch=steps_per_epoch, max_lr=lr_lists,
                    ))

            self.sched_type = sched_type
            self.sched = getattr(torch.optim.lr_scheduler, sched_type)(
                self.optim, **sched_kwargs)

            if getattr(self.opts, 'resume_train_iter', False):
                # ugly but works
                for _ in range(self.opts.resume_train_iter):
                    self.optim.step()
                    self.sched.step()

            info = info + \
                ' (' + ', '.join([f'{k}={v}' for k,
                                  v in sched_kwargs.items()]) + ')'
            log.info(info)

    def restore_checkpoint(self):
        epoch_start, iter_start = 0, 0
        if self.opts.resume:
            log.info("resuming from previous checkpoint...")
            ckpt_path = os.path.join(
                self.opts.output_path, 'models', 'latest.pth')
            if not os.path.isfile(ckpt_path):
                log.warn(f"can NOT find previous checkpoints at {ckpt_path}")
                log.warn("start training from scratch.")
            else:
                optims_scheds = {x: getattr(self, x) for x in [
                    'optim', 'sched'] if hasattr(self, x)}
                epoch_start, iter_start = utils.restore_checkpoint(self.model, ckpt_path=ckpt_path,
                                                                   device=self.opts.device, log=log, resume=True,
                                                                   optims_scheds=optims_scheds,
                                                                   no_strict_load=getattr(
                                                                       self.opts, 'no_strict_load', False),
                                                                   )
        elif self.opts.load is not None:
            log.info("loading weights from checkpoint {}...".format(self.opts.load))
            epoch_start, iter_start = utils.restore_checkpoint(
                self.model, ckpt_path=self.opts.load, device=self.opts.device, log=log,
                no_strict_load=getattr(self.opts, 'no_strict_load', False),
            )
        else:
            log.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

        if getattr(self.opts, 'coarse_init_fine', False):
            assert self.opts.load is not None
            # training fine nerf with pretrained coarse nerf as init
            log.info('initializing fine encoder with pretrained coarse encoder')
            self.model.feat_enc_fine.load_state_dict(
                self.model.feat_enc.state_dict(), strict=True)

        if getattr(self.opts, 'load_coarse', False):
            log.info('load coarse pretrained model')
            checkpoint = torch.load(self.opts.load_coarse)['model']
            # only partially load the coarse model
            self.model.load_state_dict(checkpoint, strict=False)

        if getattr(self.opts, 'resume_train_iter', False):
            assert getattr(self.opts, 'resume_train_epoch', False)
            self.epoch_start = self.opts.resume_train_epoch
            self.iter_start = self.opts.resume_train_iter

    def setup_visualizer(self):
        log.info("setting up visualizers...")
        if self.opts.tb and self.opts.local_rank == 0:
            from torch.utils import tensorboard
            self.tb = tensorboard.SummaryWriter(
                log_dir=self.opts.output_path, flush_secs=10)

    def train_model(self):
        # before training
        log.title("TRAINING START")
        self.send_results("TRAINING START", log_msg=False)
        self.timer = edict(start=time.time(), it_mean=None)
        self.it = self.iter_start
        self.val_it = math.ceil(self.opts.freq.val_it * len(self.train_loader)
                                ) if self.opts.freq.val_it > 0 else self.opts.freq.val_it
        self.ckpt_it = math.ceil(self.opts.freq.ckpt_it * len(self.train_loader)
                                 ) if self.opts.freq.ckpt_it > 0 else self.opts.freq.ckpt_it

        # training
        # resume at the end of one epoch or middle
        if self.epoch_start == 0:
            epoch_start = 0
        else:
            if self.epoch_start * len(self.train_loader) > self.iter_start:
                epoch_start = self.epoch_start - 1
            else:
                epoch_start = self.epoch_start

        for self.ep in range(epoch_start, self.opts.max_epoch):
            if self.opts.dist:
                self.train_sampler.set_epoch(self.ep)

            self.train_epoch()

            if getattr(self.opts, 'stop_epoch', False):
                if self.ep >= self.opts.stop_epoch - 1:
                    break

        # after training
        if self.opts.local_rank == 0 and self.opts.tb:
            self.tb.flush()
            self.tb.close()
        log.title("TRAINING DONE")
        self.send_results("TRAINING DONE", reset_status=True, log_msg=False)

    def train_epoch(self):
        # before train epoch
        if self.opts.dist:
            self.model_ddp.train()
        else:
            self.model.train()
        # train epoch
        tqdm_bar = tqdm.tqdm(
            self.train_loader, desc="training epoch {}".format(self.ep + 1), leave=False)
        for batch_idx, batch in enumerate(tqdm_bar):
            # train iteration
            if self.opts.resume and self.ep * len(self.train_loader) + batch_idx < self.iter_start:
                continue

            if getattr(self.opts, 'mix_data_train', False) or hasattr(self.opts, 'max_crop_height') or hasattr(self.opts, 'max_crop_width'):
                # random crop due to different image resolutions
                # print(batch['images'].shape)  # [B, V, 3, H, W], B=1

                # random crop
                ori_h, ori_w = batch['images'].shape[3:]
                assert hasattr(self.opts, 'max_crop_height')
                assert hasattr(self.opts, 'max_crop_width')
                max_crop_height = getattr(self.opts, 'max_crop_height')
                max_crop_width = getattr(self.opts, 'max_crop_width')

                # nearest size to be divisable by 16 or 32
                resize_factor = getattr(self.opts, 'resize_factor', 16)
                near_h = ori_h // resize_factor * resize_factor
                near_w = ori_w // resize_factor * resize_factor

                landscape = ori_w > ori_h

                if landscape:
                    crop_h = min(near_h, max_crop_height)
                    crop_w = min(near_w, max_crop_width)
                else:
                    crop_h = min(near_h, max_crop_width)
                    crop_w = min(near_w, max_crop_height)

                batch = RandomCrop(crop_h, crop_w, with_batch_dim=True)(batch)

            var = edict(batch)
            var = utils.move_to_device(var, self.opts.device)

            loss = self.train_iteration(var)
            tqdm_bar.set_postfix(it=self.it, loss="{:.3f}".format(loss.all))

            if self.sched_type == 'OneCycleLR':
                self.sched.step()

            # after train epoch
            if batch_idx == len(self.train_loader) - 1:
                lr_dict = self.get_cur_lrates()
                log.loss_train(self.opts, self.ep + 1,
                               lr_dict, loss.all, self.timer)

        if self.sched_type is not None and self.sched_type != 'OneCycleLR':
            self.sched.step()
        if self.opts.freq.test_ep > 0 and (self.ep + 1) % self.opts.freq.test_ep == 0:
            if getattr(self.opts, 'test_rank0', False):
                if self.opts.local_rank == 0:
                    self.test_model(ep=self.ep + 1, send_log=True, save_images=False,
                                    with_depth_metric=getattr(self.opts, 'with_depth_metric', False))
            else:
                self.test_model(ep=self.ep + 1, send_log=True, save_images=False,
                                with_depth_metric=getattr(self.opts, 'with_depth_metric', False))
        if self.opts.freq.ckpt_ep > 0 and (self.ep + 1) % self.opts.freq.ckpt_ep == 0 and self.opts.local_rank == 0:
            self.save_checkpoint(ep=self.ep + 1, it=self.it, backup_ckpt=True)

    def train_iteration(self, var):
        # before train iteration
        self.timer.it_start = time.time()

        # train iteration
        self.optim.zero_grad()

        if self.opts.dist:
            var_pred = self.model_ddp(var, mode="train")
        else:
            var_pred = self.model(var, mode="train")

        loss = self.compute_loss(var_pred, var, mode="train")
        loss = summarize_loss(loss, self.opts.loss_weight)

        if isinstance(loss.all, float):
            print('float loss, skip backward')
            return loss

        if torch.isnan(loss.all):
            print('nan loss, skip backward')
            return loss

        loss.all.backward()

        if self.opts.optim.clip_enc is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.feat_enc.parameters(), self.opts.optim.clip_enc)

        self.optim.step()

        # after train iteration
        self.it += 1
        self.timer.it_end = time.time()
        utils.update_timer(self.opts, self.timer, self.ep,
                           len(self.train_loader))
        if self.opts.freq.scalar > 0 and self.it % self.opts.freq.scalar == 0:
            cur_lrates = self.get_cur_lrates()
            self.log_scalars(loss, self.opts.loss_weight,
                             lrates=cur_lrates, step=self.it, split="train")
        if self.ckpt_it > 0 and self.it % self.ckpt_it == 0 and self.opts.local_rank == 0:
            self.save_checkpoint(ep=self.ep + 1, it=self.it,
                                 backup_ckpt=getattr(
                                     self.opts, 'backup_ckpt', False),
                                 backup_latest_ckpt=getattr(self.opts.freq, 'latest_ckpt_ep', False) and (
                                     (self.ep + 1) % self.opts.freq.latest_ckpt_ep == 0)
                                 )

        if getattr(self.opts.freq, 'latest_ckpt_iter', False) and self.opts.local_rank == 0:
            self.save_checkpoint(ep=self.ep + 1, it=self.it,
                                 backup_ckpt=False,
                                 backup_latest_ckpt_iter=self.it % self.opts.freq.latest_ckpt_iter == 0,
                                 )

        if self.val_it > 0 and self.it % self.val_it == 0 and hasattr(self, 'val_loader') and self.opts.local_rank == 0:
            self.validate_model(iter=self.it,
                                with_depth_metric=getattr(self.opts, 'with_depth_metric', False))

        # log training images to get some sense of the training data
        if getattr(self.opts, 'log_train_imgs', False) and self.opts.freq.scalar > 0 and self.it % self.opts.freq.scalar == 0 and self.opts.local_rank == 0:
            # log pred and gt images
            batch_size = var['images'].shape[0]
            img_hw = var['img_wh'][0].cpu().numpy().tolist()[::-1]
            pred_rgb = var_pred.rgb.reshape(batch_size, *img_hw, -1)

            pred_rgb_tb = pred_rgb.permute(0, 3, 1, 2)  # [B, 3, H, W]

            target_gt_tb = var.images[:, -1]  # [B, 3, H, W]
            img_pred_gt = torch.cat((pred_rgb_tb, target_gt_tb), dim=-1)

            self.tb.add_image('0train/img_pred_gt',
                              img_pred_gt[0], self.it)  # [3, H, W*2]

            # log input views
            b, v, _, h, w = var['images'].size()
            input_rgbs = var['images'].permute(0, 2, 3, 4, 1).reshape(
                b, 3, h, w, v).to(pred_rgb_tb.device)  # [B, 3, H, W*(V+1)]
            img_list = [rgb[:, :, :, :, 0]
                        for rgb in input_rgbs.chunk(v, dim=-1)]
            # [B, 3, H, W*(V-1)], source image
            img_input = torch.cat(img_list[:-1], dim=-1)
            self.tb.add_image('0train/img_input',
                              img_input[0], self.it)  # [3, H, W*(V-1)]

            # log depth
            if getattr(self.opts, 'log_train_depth', False):
                pred_depth = var_pred['depth'].reshape(
                    batch_size, *img_hw).detach()  # [B, H, W]

                # vis depth (inverse depth)
                pred_depth_viz = viz_depth_tensor(
                    1. / pred_depth[0].cpu()).float() / 255.  # [3, H, W]
                self.tb.add_image('0train/depth_pred', pred_depth_viz, self.it)

        # check gpu
        if self.it == 5 and self.opts.local_rank == 0:
            os.system('nvidia-smi')

        return loss

    def compute_loss(self, pred, src, mode=None):
        self.n_src_views = src.images.size(1) - 1

        loss = edict()
        batch_size, n_views, n_chnl = src.images.shape[:3]
        assert n_views == (
            self.n_src_views + 1), "Make sure the last views are provided as the GT target view"
        # (b, h*w, 3)
        target_gt = src.images[:, -
                               1].reshape(batch_size, n_chnl, -1).permute(0, 2, 1)

        if getattr(self.opts.nerf, f"rand_rays_{mode}", False) and mode in ["train", "test-optim"] and not getattr(self.opts, 'radiance_downsample_factor', False):
            target_gt = target_gt[:, pred.ray_idx]

        # compute image losses
        if self.opts.loss_weight.render is not None:
            loss.render = self.L1_loss(pred.rgb, target_gt)

        # ssim loss
        if getattr(self.opts.loss_weight, 'ssim', False):
            tmp_img_gt = target_gt
            tmp_img_pred = pred.rgb

            img_hw = src['img_wh'][0].cpu().numpy().tolist()[::-1]
            curr_h, curr_w = img_hw

            assert target_gt.size(1) == curr_h * curr_w
            assert pred.rgb.size(1) == curr_h * curr_w

            tmp_img_gt = tmp_img_gt.view(batch_size, curr_h, curr_w, 3).permute(
                0, 3, 1, 2)  # [B, 3, H, W]
            tmp_img_pred = tmp_img_pred.view(
                batch_size, curr_h, curr_w, 3).permute(0, 3, 1, 2)

            loss.ssim = self.ssim_loss_func(tmp_img_gt, tmp_img_pred).mean()

        # lpips loss
        if getattr(self.opts.loss_weight, 'lpips', False):
            tmp_img_gt = target_gt
            tmp_img_pred = pred.rgb

            img_hw = src['img_wh'][0].cpu().numpy().tolist()[::-1]
            curr_h, curr_w = img_hw

            assert target_gt.size(1) == curr_h * curr_w
            assert pred.rgb.size(1) == curr_h * curr_w

            tmp_img_gt = tmp_img_gt.view(batch_size, curr_h, curr_w, 3).permute(
                0, 3, 1, 2)  # [B, 3, H, W]
            tmp_img_pred = tmp_img_pred.view(
                batch_size, curr_h, curr_w, 3).permute(0, 3, 1, 2)

            # images must be in [-1, 1] for computing lpips loss
            tmp_img_gt = tmp_img_gt * 2 - 1
            tmp_img_pred = tmp_img_pred * 2 - 1
            # assert tmp_img_gt.min() >= -1 and tmp_img_gt.max() <= 1 and tmp_img_pred.min() >= -1 and tmp_img_pred.max() <= 1

            loss.lpips = self.lpips_loss_func(tmp_img_gt, tmp_img_pred).mean()

        return loss

    @torch.no_grad()
    def log_scalars(self, loss=None, loss_weight=None, metric=None, lrates=None, step=0, split="train"):
        if loss is not None:
            for key, value in loss.items():
                if key == "all":
                    continue
                if loss_weight[key] is not None and self.opts.local_rank == 0:
                    self.tb.add_scalar(
                        "{0}/loss_{1}".format(split, key), value, step)
        if metric is not None and self.opts.local_rank == 0:
            for key, value in metric.items():
                mean_value = np.array(value).mean()
                self.tb.add_scalar(
                    "{0}/{1}".format(split, key), mean_value, step)
        if lrates is not None and self.opts.local_rank == 0:
            for key, value in lrates.items():
                self.tb.add_scalar("{0}/{1}".format('lrate', key), value, step)

    @torch.no_grad()
    def get_cur_lrates(self):
        if self.opts.optim.sched:
            if getattr(self.opts, 'freeze_encoder', False) or getattr(self.opts, 'freeze_coarse_nerf', False):
                lr_enc = 0.
                lr_dec = self.sched.get_last_lr()[0]
            else:
                lr_enc, lr_dec = self.sched.get_last_lr()[:2]
        else:
            lr_enc = self.opts.optim.lr_enc
            lr_dec = self.opts.optim.lr_dec
        lr_dict = dict(enc=lr_enc, dec=lr_dec)
        if self.opts.nerf.fine_sampling:
            lr_dict['dec_fine'] = lr_dec
        return lr_dict

    def save_checkpoint(self, ep=0, it=0, backup_ckpt=True, backup_latest_ckpt=False, backup_latest_ckpt_iter=False):
        save_train_info = True

        checkpoint = dict(model=self.model.state_dict())
        if save_train_info:
            train_info = dict(optim=self.optim.state_dict())
            if self.sched_type is not None:
                train_info.update(dict(sched=self.sched.state_dict()))
            checkpoint.update(train_info)

        utils.save_checkpoint(self.opts.output_path,
                              checkpoint, ep=ep, it=it,
                              backup_ckpt=backup_ckpt,
                              backup_latest_ckpt=backup_latest_ckpt,
                              backup_latest_ckpt_iter=backup_latest_ckpt_iter,
                              )

    def send_results(self, msg, reset_status=False, log_msg=True):
        if log_msg:
            log.metric_test(re.sub('<[^<]+?>', '', msg.split('\n')[-1]))

    @torch.no_grad()
    def validate_model(self, iter=None,
                       with_depth_metric=False,
                       ):
        assert hasattr(self, 'val_loader'), "please load validation dataset."
        self.model.eval()
        data_outdir = os.path.join(self.opts.output_path, 'validation')
        os.makedirs(data_outdir, exist_ok=True)
        eval_tools = EvalTools(device=self.opts.device)
        metrics_dict = {k: [] for k in eval_tools.support_metrics}

        if not with_depth_metric:
            del metrics_dict['depth_abs']

        # 5 validation samples
        tqdm_loader = tqdm.tqdm(
            self.val_loader, desc="validating", leave=False)
        for batch_idx, batch in enumerate(tqdm_loader):
            if iter == 0 and batch_idx > 0:
                break
            var = edict(batch)
            batch_size = var.images.shape[0]
            var = utils.move_to_device(var, self.opts.device)

            # nearest size to be divisable by 16 or 32
            resize_factor = getattr(self.opts, 'resize_factor', 16)
            if var.images.shape[3] % resize_factor != 0 or var.images.shape[4] % resize_factor != 0:
                near_inference_size = [var.images.shape[3] // resize_factor *
                                       resize_factor, var.images.shape[4] // resize_factor * resize_factor]
                # print(near_inference_size)
            else:
                near_inference_size = None

            if near_inference_size is not None:
                inference_size = near_inference_size

                ori_size = var.images.shape[3:]
                scale_factor_y = inference_size[0] / ori_size[0]
                scale_factor_x = inference_size[1] / ori_size[1]

                batch_size = var.images.shape[0]
                num_views = var.images.shape[1]

                ori_images = var.images.clone()

                tmp_imgs = var.images.view(-1, 3, *ori_size)  # [B*V, 3, H, W]
                tmp_imgs = F.interpolate(
                    tmp_imgs, size=inference_size, mode='bilinear', align_corners=True)
                var.images = tmp_imgs.view(
                    batch_size, num_views, 3, *inference_size)

                # update intrinsics
                intrinsic = var.intrinsics.clone()  # [B, V, 3, 3]
                intrinsic[:, :, :1] = intrinsic[:, :, :1] * scale_factor_x
                intrinsic[:, :, 1:2] = intrinsic[:, :, 1:2] * scale_factor_y
                var.intrinsics = intrinsic

                # update size
                ori_wh = var.img_wh.clone()
                var.img_wh[:, 0] = inference_size[1]
                var.img_wh[:, 1] = inference_size[0]

            var = self.model(var, mode="val")

            # resize back
            if near_inference_size is not None:
                # resize rgb
                var.images = ori_images
                var.img_wh = ori_wh

                tmp_rgb = var.rgb.reshape(
                    batch_size, *inference_size, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
                tmp_rgb = F.interpolate(
                    tmp_rgb, size=ori_size, mode='bilinear', align_corners=True)
                var.rgb = tmp_rgb.view(
                    batch_size, 3, -1).permute(0, 2, 1)  # [B, H*W, 3]

                # resize depth
                tmp_depth = var.depth.reshape(
                    batch_size, *inference_size, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]
                tmp_depth = F.interpolate(
                    tmp_depth, size=ori_size, mode='bilinear', align_corners=True)
                var.depth = tmp_depth.view(batch_size, -1)  # [B, H*W]

                batch['img_wh'] = ori_wh.cpu()

            # save image
            img_hw = batch['img_wh'][0].numpy().tolist()[::-1]
            pred_rgb = var['rgb'].reshape(
                batch_size, *img_hw, -1)  # [B, H, W, 3]

            for batch_idx, cur_rgb in enumerate(pred_rgb):
                pred_rgb_nb = cur_rgb.detach().cpu().numpy()
                # h,w,3
                gt_rgb_nb = var.images[batch_idx, -
                                       1].permute(1, 2, 0).detach().cpu().numpy()

                if 'dtu' == self.val_loader.dataset.get_name():
                    assert 'depth' in batch, "Must provide 'depth' of target view for validation"
                    depth = batch['depth'][batch_idx].detach().cpu().numpy()
                    image_mask = depth == 0
                elif 'mask' in batch:  # regnerf evaluation
                    # 0 is background
                    image_mask = batch['mask'][batch_idx] == 0
                else:
                    image_mask = None

                eval_tools.set_inputs(pred_rgb_nb, gt_rgb_nb, image_mask,
                                      pred_depth=var['depth'].reshape(batch_size, *img_hw)[batch_idx].detach(
                                      ).cpu().numpy() if with_depth_metric and 'depth' in var else None,
                                      gt_depth=var['depth_gt'].reshape(batch_size, *img_hw)[batch_idx].detach(
                                      ).cpu().numpy() if with_depth_metric and 'depth_gt' in var else None,
                                      )
                cur_metrics = eval_tools.get_metrics()
                for k, v in cur_metrics.items():
                    metrics_dict[k].append(v)

        self.log_scalars(metric=metrics_dict, step=iter, split="val")

        if self.opts.dist:
            self.model_ddp.train()
        else:
            self.model.train()

    @torch.no_grad()
    def test_model(self, ep=None, send_log=True, save_images=False, leave_tqdm=False,
                   save_depth=True,
                   save_gt_depth=True,
                   save_depth_np=False,
                   save_gt_depth_np=False,
                   with_depth_metric=False,
                   test_on_val_set=False,
                   ):
        if test_on_val_set:
            assert hasattr(
                self, 'val_loader'), "Must load the val data for testing."
            test_loaders = [self.val_loader]
        else:
            assert hasattr(
                self, 'test_loaders'), "Must load the test data for testing."
            test_loaders = self.test_loaders

        test_outroot = os.path.join(self.opts.output_path, 'test')
        os.makedirs(test_outroot, exist_ok=True)
        eval_tools = EvalTools(device=self.opts.device)
        metrics_dict = {}
        metrics_list_dict = {}

        self.model.eval()
        for data_loader in test_loaders:
            dataname = data_loader.dataset.get_name()
            metrics_dict[dataname] = OrderedDict()
            metrics_list_dict[dataname] = []
            data_outdir = os.path.join(test_outroot, dataname)

            if getattr(self.opts, 'save_name_suffix', None):
                data_outdir = data_outdir + '_' + self.opts.save_name_suffix

            os.makedirs(data_outdir, exist_ok=True)

            # tensorboard summary images
            num_summary_images = 8
            sample_interval = len(data_loader) // num_summary_images if len(
                data_loader) % num_summary_images == 0 else len(data_loader) // num_summary_images + 1

            count = 0

            tqdm_desc = f"testing {dataname}" if ep is None else f"testing {dataname} [epoch {ep}]"
            for batch_idx, batch in enumerate(tqdm.tqdm(data_loader, desc=tqdm_desc, leave=leave_tqdm)):
                if hasattr(self, "it") and self.it == 0 and batch_idx > 0:
                    break

                var = edict(batch)
                var = utils.move_to_device(var, self.opts.device)

                # nearest size to be divisable by 16 or 32
                resize_factor = getattr(self.opts, 'resize_factor', 16)
                if var.images.shape[3] % resize_factor != 0 or var.images.shape[4] % resize_factor != 0:
                    near_inference_size = [var.images.shape[3] // resize_factor *
                                           resize_factor, var.images.shape[4] // resize_factor * resize_factor]
                else:
                    near_inference_size = None

                # resize then inference
                if getattr(self.opts, 'inference_size', False) or near_inference_size is not None:

                    if getattr(self.opts, 'llff_inference_size', False) and dataname == 'llff':
                        inference_size = self.opts.llff_inference_size
                    elif getattr(self.opts, 'blender_inference_size', False) and dataname == 'blender':
                        inference_size = self.opts.blender_inference_size
                    else:
                        inference_size = self.opts.inference_size if getattr(
                            self.opts, 'inference_size', False) else near_inference_size

                    ori_size = var.images.shape[3:]
                    scale_factor_y = inference_size[0] / ori_size[0]
                    scale_factor_x = inference_size[1] / ori_size[1]

                    batch_size = var.images.shape[0]
                    num_views = var.images.shape[1]

                    ori_images = var.images.clone()

                    # [B*V, 3, H, W]
                    tmp_imgs = var.images.view(-1, 3, *ori_size)
                    tmp_imgs = F.interpolate(
                        tmp_imgs, size=inference_size, mode='bilinear', align_corners=True)
                    var.images = tmp_imgs.view(
                        batch_size, num_views, 3, *inference_size)

                    # update intrinsics
                    intrinsic = var.intrinsics.clone()  # [B, V, 3, 3]
                    intrinsic[:, :, :1] = intrinsic[:, :, :1] * scale_factor_x
                    intrinsic[:, :, 1:2] = intrinsic[:,
                                                     :, 1:2] * scale_factor_y
                    var.intrinsics = intrinsic

                    # update size
                    ori_wh = var.img_wh.clone()
                    var.img_wh[:, 0] = inference_size[1]
                    var.img_wh[:, 1] = inference_size[0]
                
                var = self.model(var, mode="test")

                if var['rgb'].isnan().any():
                    print('pred nan')
                    print(var['rgb'].isnan().sum())
                    var['rgb'] = torch.nan_to_num(var['rgb'], 0.)

                if var['images'].isnan().any():
                    print('ori image nan')

                # resize back
                if not getattr(self.opts, 'no_resize_back', False) and (getattr(self.opts, 'inference_size', False) or near_inference_size is not None):
                    # resize rgb
                    var.images = ori_images
                    var.img_wh = ori_wh

                    tmp_rgb = var.rgb.reshape(
                        batch_size, *inference_size, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
                    tmp_rgb = F.interpolate(
                        tmp_rgb, size=ori_size, mode='bilinear', align_corners=True)
                    var.rgb = tmp_rgb.view(
                        batch_size, 3, -1).permute(0, 2, 1)  # [B, H*W, 3]

                    # resize depth
                    tmp_depth = var.depth.reshape(
                        batch_size, *inference_size, 1).permute(0, 3, 1, 2)  # [B, 1, H, W]
                    tmp_depth = F.interpolate(
                        tmp_depth, size=ori_size, mode='bilinear', align_corners=True)
                    var.depth = tmp_depth.view(batch_size, -1)  # [B, H*W]

                    batch['img_wh'] = ori_wh.cpu()

                if getattr(self.opts, 'no_resize_back', False):
                    batch['img_wh'][0, 0] = self.opts.inference_size[1]
                    batch['img_wh'][0, 1] = self.opts.inference_size[0]

                # save image
                batch_size = var['images'].shape[0]
                img_hw = batch['img_wh'][0].numpy().tolist()[::-1]
                pred_rgb = var['rgb'].reshape(batch_size, *img_hw, -1)

                # tensorboard summary
                if hasattr(self, 'tb') and batch_idx % sample_interval == 0:
                    pred_rgb_tb = pred_rgb.permute(0, 3, 1, 2)  # [B, 3, H, W]
                    b, _, h, w = pred_rgb_tb.shape
                    v = batch['images'].size(1)
                    input_rgbs = batch['images'].permute(0, 2, 3, 4, 1).reshape(
                        b, 3, h, w, v).to(pred_rgb_tb.device)  # [B, 3, H, W*(V+1)]
                    img_list = [rgb[:, :, :, :, 0]
                                for rgb in input_rgbs.chunk(v, dim=-1)]
                    # [B, 3, H, W*(V-1)], source image
                    img_input = torch.cat(img_list[:-1], dim=-1)
                    # [B, 3, H, W*2], pred and gt
                    img_pred_gt = torch.cat(
                        [pred_rgb_tb, img_list[-1]], dim=-1)
                    img_error = compute_image_diff(
                        pred_rgb_tb, batch['images'][:, -1].to(pred_rgb_tb.device))  # [B, H, W]
                    img_pred_gt_error = torch.cat((img_pred_gt, img_error.unsqueeze(
                        1).repeat(1, 3, 1, 1)), dim=-1)  # [B, 3, H, W*3]

                    self.tb.add_image('%s_%d/0_img_pred_gt_error' % (
                        dataname, batch_idx // sample_interval), img_pred_gt_error[0], ep)  # [3, H, W*3]
                    self.tb.add_image('%s_%d/1_img_input' % (dataname, batch_idx //
                                      sample_interval), img_input[0], ep)  # [3, H, W*(V+2)]

                    # summary depth if have
                    if 'depth' in batch:
                        pred_depth = var['depth'].reshape(
                            batch_size, *img_hw)  # [B, H, W]
                        gt_depth = batch['depth'].to(
                            pred_depth.device)  # [B, H, W]

                        # vis depth
                        pred_depth_viz = viz_depth_tensor(
                            pred_depth[0].cpu())  # [3, H, W]
                        gt_depth_viz = viz_depth_tensor(gt_depth[0].cpu())
                        depth_pred_gt = torch.cat(
                            (pred_depth_viz, gt_depth_viz), dim=-1)  # [3, H, W*2]

                        # vis depth error
                        depth_error = compute_depth_diff(pred_depth, gt_depth, valid_mask=gt_depth > 0.,
                                                         min_depth=batch['near_fars'][0, -1,
                                                                                      0], max_depth=batch['near_fars'][0, -1, 1],
                                                         ).cpu()

                        depth_pred_gt_error = torch.cat((depth_pred_gt.float(
                        ) / 255., depth_error.unsqueeze(1).repeat(1, 3, 1, 1)[0]), dim=-1)

                        self.tb.add_image('%s_%d/2_depth_pred_gt_error' % (
                            dataname, batch_idx // sample_interval), depth_pred_gt_error, ep)

                    # log depth prediction
                    if getattr(self.opts, 'log_test_depth', False):
                        pred_depth = var['depth'].reshape(
                            batch_size, *img_hw).detach()  # [B, H, W]

                        # vis depth (inverse depth)
                        pred_depth_viz = viz_depth_tensor(
                            1. / pred_depth[0].cpu()).float() / 255.  # [3, H, W]
                        self.tb.add_image(
                            '%s_%d/2_depth_pred' % (dataname, batch_idx // sample_interval), pred_depth_viz, ep)

                if save_images and save_depth and 'depth' in var:
                    pred_depth = var['depth'].reshape(
                        batch_size, *img_hw)  # [B, H, W]

                if save_gt_depth and 'depth_gt' in var:
                    gt_depth = var['depth_gt'].reshape(batch_size, *img_hw)

                if save_images:
                    for batch_idx, cur_rgb in enumerate(pred_rgb):
                        pred_rgb_nb = (
                            cur_rgb.detach().cpu().numpy() * 255).astype('uint8')

                        if 'realestate' in dataname or getattr(self.opts, 'filename_with_number', False):
                            # no scene and view id info, or for simplicity
                            out_name = '%04d_target_pred.png' % count
                        elif 'mipnerf360' in dataname or 'ibrnet_llff_test' in dataname:
                            out_name = f"{batch['scene'][batch_idx]}_view{count:03d}_target_pred.png"
                        else:
                            src_ids_str = '_'.join(
                                [f'{x:02d}' for x in batch['view_ids'][batch_idx][:self.n_src_views]])
                            # out_name = f"{batch['scene'][batch_idx]}_view{batch['view_ids'][batch_idx][-1]:02d}_src{src_ids_str}.png"
                            if getattr(self.opts, 'save_source_view_id', False):
                                src_ids_str = '_'.join(
                                    [f'{x:02d}' for x in batch['view_ids'][batch_idx][:self.n_src_views]])
                                out_name = f"{batch['scene'][batch_idx]}_view{batch['view_ids'][batch_idx][-1]:02d}_target_pred_src{src_ids_str}.png"
                            else:
                                out_name = f"{batch['scene'][batch_idx]}_view{batch['view_ids'][batch_idx][-1]:02d}_target_pred.png"
                            if ep is not None:
                                out_name = f"ep{ep}_{out_name}"

                        imageio.imwrite(os.path.join(
                            data_outdir, out_name), pred_rgb_nb)

                        if save_depth:
                            pred_depth_np = pred_depth[batch_idx].detach(
                            ).cpu().numpy()

                            if save_depth_np:
                                # save original value
                                depth_out_name = out_name[:-4] + '_depth.npy'
                                np.save(os.path.join(data_outdir,
                                        depth_out_name), pred_depth_np)

                            pred_depth_np[pred_depth_np ==
                                          0] = 99999.  # very large value for zero
                            pred_depth_vis = utils.viz_depth_np(
                                1. / pred_depth_np)
                            depth_out_name = out_name[:-4] + '_depth.png'
                            imageio.imwrite(os.path.join(
                                data_outdir, depth_out_name), pred_depth_vis)

                        if save_gt_depth:
                            gt_depth_np = gt_depth[batch_idx].detach(
                            ).cpu().numpy()

                            if save_gt_depth_np:
                                # save original value
                                depth_out_name = out_name[:-
                                                          4] + '_depth_gt.npy'
                                np.save(os.path.join(data_outdir,
                                        depth_out_name), gt_depth_np)

                            gt_depth_np[gt_depth_np ==
                                        0] = 99999.  # very large value for zero
                            gt_depth_vis = utils.viz_depth_np(1. / gt_depth_np)
                            depth_out_name = out_name[:-4] + '_depth_gt.png'
                            imageio.imwrite(os.path.join(
                                data_outdir, depth_out_name), gt_depth_vis)

                # save input source and target images
                if getattr(self.opts, 'save_source_target_images', False):
                    source_target = var['images'][0].permute(
                        0, 2, 3, 1).cpu().numpy()  # [(V+1), H, W, 3]
                    total_views = source_target.shape[0]
                    for i in range(total_views):
                        if getattr(self.opts, 'filename_with_number', False):
                            filename = ('%04d_source_%d.png' % (
                                count, i)) if i < total_views - 1 else ('%04d_target_gt.png' % count)
                        else:
                            if i == total_views - 1:
                                if 'mipnerf360' in dataname or 'ibrnet_llff_test' in dataname:
                                    filename = f"{batch['scene'][0]}_view{count:03d}_target_gt.png"
                                else:
                                    filename = f"{batch['scene'][0]}_view{batch['view_ids'][0][-1]:02d}_target_gt.png"
                            else:
                                if 'mipnerf360' in dataname or 'ibrnet_llff_test' in dataname:
                                    filename = f"{batch['scene'][0]}_view{count:03d}_source_{i:02d}.png"
                                else:
                                    filename = f"{batch['scene'][0]}_view{batch['view_ids'][0][-1]:02d}_source_{i:02d}.png"
                        filename = os.path.join(data_outdir, filename)
                        imageio.imwrite(
                            filename, (source_target[i] * 255.).astype('uint8'))

                if 'realestate' in dataname or getattr(self.opts, 'filename_with_number', False) or 'mipnerf360' in dataname or 'ibrnet_llff_test' in dataname:
                    count += 1

                # log metric
                if self.opts.local_rank == 0:
                    for batch_idx, cur_rgb in enumerate(pred_rgb):
                        pred_rgb_nb = cur_rgb.detach().cpu().numpy()
                        # h,w,3
                        gt_rgb_nb = var.images[batch_idx, -
                                               1].permute(1, 2, 0).detach().cpu().numpy()

                        # image_mask is invalid mask
                        if 'depth' in batch and not getattr(self.opts, 'eval_no_depth_mask', False):
                            depth = batch['depth'][batch_idx].detach(
                            ).cpu().numpy()
                            image_mask = depth == 0
                        elif 'mask' in batch:  # regnerf evaluation
                            # 0 is background
                            image_mask = batch['mask'][batch_idx] == 0
                        else:
                            image_mask = None

                        eval_tools.set_inputs(pred_rgb_nb, gt_rgb_nb, image_mask,
                                              full_img_eval=getattr(
                                                  self.opts, 'full_img_eval', False),
                                              pred_depth=var['depth'].reshape(batch_size, *img_hw)[batch_idx].detach(
                                              ).cpu().numpy() if with_depth_metric and 'depth' in var else None,
                                              gt_depth=var['depth_gt'].reshape(batch_size, *img_hw)[batch_idx].detach(
                                              ).cpu().numpy() if with_depth_metric and 'depth_gt' in var else None,
                                              )
                        if getattr(self.opts, 'data_test', False):
                            report_full_scores = getattr(
                                getattr(self.opts.data_test, dataname), "report_full_scores", False)
                        else:
                            report_full_scores = False
                        cur_metrics = eval_tools.get_metrics(
                            return_full=report_full_scores)
                        metrics_list_dict[dataname].append(cur_metrics)
                        metrics_dict[dataname][f"{var.scene[batch_idx]}_{var.view_ids[batch_idx, -1]:03d}"] = cur_metrics

            # reset params
            self.model.nerf_setbg_opaque = False

        # per scene per view summary
        # sum_dict = summarize_metrics(metrics_dict, test_outroot, ep=ep)
        # only all mean summary
        if self.opts.local_rank == 0:
            sum_dict = summarize_metrics_list(
                metrics_list_dict, test_outroot, ep=ep)
        if send_log and self.opts.local_rank == 0:
            tg_msg = f"{self.ep:02d}, {self.it:06d};" if hasattr(
                self, 'ep') and hasattr(self, 'it') else ""
            for cur_dataname, cur_datametric in sum_dict.items():
                metric_avg = {k: np.array(v).mean()
                              for k, v in cur_datametric.items()}

                if 'depth_abs' in metric_avg and 'thres005' in metric_avg and 'thres001' in metric_avg:
                    tg_msg = tg_msg + \
                        f" {cur_dataname.upper()[0]}: {metric_avg['PSNR']:.2f}, {metric_avg['SSIM']:.3f}," + \
                            f" {metric_avg['LPIPS']:.3f}, {metric_avg['depth_abs']:.3f}, {metric_avg['thres005']:.3f}, {metric_avg['thres001']:.3f},"
                else:
                    tg_msg = tg_msg + \
                        f" {cur_dataname.upper()[0]}: {metric_avg['PSNR']:.2f}, {metric_avg['SSIM']:.3f}, {metric_avg['LPIPS']:.3f},"
                if hasattr(self, 'tb'):
                    self.log_scalars(metric=metric_avg,
                                     step=ep, split=cur_dataname)
            self.send_results(tg_msg)

        if self.opts.dist:
            self.model_ddp.train()
        else:
            self.model.train()

    @torch.no_grad()
    def test_model_video(self, ep=None, leave_tqdm=False,
                         save_depth_video=False,
                         save_depth_np=False,
                         ):
        assert hasattr(
            self, 'test_loaders'), "Must load the test data for testing."
        test_outroot = os.path.join(self.opts.output_path, 'test_videos')
        os.makedirs(test_outroot, exist_ok=True)

        self.model.eval()
        for data_loader in self.test_loaders:
            dataname = data_loader.dataset.get_name()
            data_outdir = os.path.join(test_outroot, dataname)
            if getattr(self.opts, 'save_name_suffix', None):
                data_outdir = data_outdir + '_' + self.opts.save_name_suffix
            os.makedirs(data_outdir, exist_ok=True)

            # set rendering parameters
            self.model.nerf_setbg_opaque = False
            render_path_mode = 'interpolate'

            count = 0
            tqdm_desc = f"testing {dataname}" if ep is None else f"testing {dataname} [epoch {ep}]"
            for batch in tqdm.tqdm(data_loader, desc=tqdm_desc, leave=leave_tqdm):
                var = edict(batch)
                var = utils.move_to_device(var, self.opts.device)

                # nearest size to be divisable by 16 or 32
                resize_factor = getattr(self.opts, 'resize_factor', 16)
                if var.images.shape[3] % resize_factor != 0 or var.images.shape[4] % resize_factor != 0:
                    near_inference_size = [var.images.shape[3] // resize_factor *
                                           resize_factor, var.images.shape[4] // resize_factor * resize_factor]
                    # print(near_inference_size)
                else:
                    near_inference_size = None

                # resize then inference
                if getattr(self.opts, 'inference_size', False) or near_inference_size is not None:
                    inference_size = self.opts.inference_size if getattr(
                        self.opts, 'inference_size', False) else near_inference_size
                    ori_size = var.images.shape[3:]
                    scale_factor_y = inference_size[0] / ori_size[0]
                    scale_factor_x = inference_size[1] / ori_size[1]

                    batch_size = var.images.shape[0]
                    num_views = var.images.shape[1]

                    ori_images = var.images.clone()

                    # [B*V, 3, H, W]
                    tmp_imgs = var.images.view(-1, 3, *ori_size)
                    tmp_imgs = F.interpolate(
                        tmp_imgs, size=inference_size, mode='bilinear', align_corners=True)
                    var.images = tmp_imgs.view(
                        batch_size, num_views, 3, *inference_size)

                    # update intrinsics
                    intrinsic = var.intrinsics.clone()  # [B, V, 3, 3]
                    intrinsic[:, :, :1] = intrinsic[:, :, :1] * scale_factor_x
                    intrinsic[:, :, 1:2] = intrinsic[:,
                                                     :, 1:2] * scale_factor_y
                    var.intrinsics = intrinsic

                    # update size
                    ori_wh = var.img_wh.clone()
                    var.img_wh[:, 0] = inference_size[1]
                    var.img_wh[:, 1] = inference_size[0]

                var = self.model(var, mode="test",
                                 render_video=self.opts.nerf.render_video, render_path_mode=render_path_mode)

                # resize back
                if not getattr(self.opts, 'no_resize_back', False) and (getattr(self.opts, 'inference_size', False) or near_inference_size is not None):
                    # resize rgb
                    var.images = ori_images
                    var.img_wh = ori_wh

                    if 'llff' in dataname:
                        real_frames = self.opts.nerf.video_n_frames
                    else:
                        real_frames = self.opts.nerf.video_n_frames // 3 * self.opts.n_src_views

                    tmp_rgb = var.rgb.reshape(
                        batch_size * real_frames, *inference_size, 3).permute(0, 3, 1, 2)  # [B*N, 3, H, W]
                    tmp_rgb = F.interpolate(
                        tmp_rgb, size=ori_size, mode='bilinear', align_corners=True)
                    var.rgb = tmp_rgb.view(
                        batch_size, real_frames, 3, -1).permute(0, 1, 3, 2)  # [B, N, H*W, 3]

                    # resize depth
                    tmp_depth = var.depth.reshape(
                        batch_size * real_frames, *inference_size, 1).permute(0, 3, 1, 2)  # [B*N, 1, H, W]
                    tmp_depth = F.interpolate(
                        tmp_depth, size=ori_size, mode='bilinear', align_corners=True)
                    var.depth = tmp_depth.view(
                        batch_size, real_frames, -1)  # [B, N, H*W]

                if getattr(self.opts, 'no_resize_back', False):
                    batch['img_wh'][0, 0] = near_inference_size[1] if not getattr(
                        self.opts, 'inference_size', False) else self.opts.inference_size[1]
                    batch['img_wh'][0, 1] = near_inference_size[0] if not getattr(
                        self.opts, 'inference_size', False) else self.opts.inference_size[0]

                # save videos
                batch_size = var['images'].shape[0]
                img_hw = batch['img_wh'][0].numpy().tolist()[::-1]
                if 'llff' in dataname:
                    real_frames = self.opts.nerf.video_n_frames
                else:
                    real_frames = self.opts.nerf.video_n_frames // 3 * self.opts.n_src_views
                pred_rgb = var['rgb'].reshape(
                    batch_size, real_frames, *img_hw, -1)
                for batch_idx, cur_rgb in enumerate(pred_rgb):
                    pred_rgb_nb = (cur_rgb.detach().cpu().numpy()
                                   * 255).astype('uint8')

                    if 'realestate' in dataname:
                        out_name = '%04d_pred.mp4' % count
                    elif 'mipnerf360' in dataname or 'ibrnet_llff_test' in dataname:
                        out_name = f"{batch['scene'][batch_idx]}_view{count:03d}_target_pred.mp4"
                    else:
                        src_ids_str = '_'.join(
                            [f'{x:02d}' for x in batch['view_ids'][batch_idx][:self.n_src_views]])
                        out_name = f"{batch['scene'][batch_idx]}_view{batch['view_ids'][batch_idx][-1]:02d}_src{src_ids_str}.mp4"
                    if ep is not None:
                        out_name = f"ep{ep}_{out_name}"
                    pred_rgb_nb_list = [pred_rgb_nb[x]
                                        for x in range(pred_rgb_nb.shape[0])]

                    if getattr(self.opts, 'save_video_frames', False):
                        for frame_idx in range(len(pred_rgb_nb_list)):
                            save_name = os.path.join(
                                data_outdir, out_name[:-4] + '_frame%03d.png' % frame_idx)
                            imageio.imwrite(
                                save_name, pred_rgb_nb_list[frame_idx])
                    else:
                        utils.write_video(os.path.join(
                            data_outdir, out_name), pred_rgb_nb_list)

                # save depth video
                if save_depth_video:
                    if 'llff' in dataname:
                        real_frames = self.opts.nerf.video_n_frames
                    else:
                        real_frames = self.opts.nerf.video_n_frames // 3 * self.opts.n_src_views

                    pred_depth = var['depth'].reshape(
                        batch_size, real_frames, *img_hw).detach().cpu().numpy()  # [B, N, H, W]
                    for batch_idx, cur_depth in enumerate(pred_depth):
                        if dataname == 'dtu':
                            # use dtu depth range: [2.125, 4.525]
                            pred_depth_vis = [utils.viz_depth_np(
                                1. / x, vmin=1. / 4.525, vmax=1. / 2.125) for x in cur_depth]
                        else:
                            pred_depth_vis = [utils.viz_depth_np(
                                1. / x) for x in cur_depth]

                        src_ids_str = '_'.join(
                            [f'{x:02d}' for x in batch['view_ids'][batch_idx][:self.n_src_views]])
                        if 'realestate' in dataname:
                            out_name = '%04d_pred.png' % count
                        elif 'mipnerf360' in dataname or 'ibrnet_llff_test' in dataname:
                            out_name = f"{batch['scene'][batch_idx]}_view{count:03d}_target_pred.png"
                        else:
                            out_name = f"{batch['scene'][batch_idx]}_view{batch['view_ids'][batch_idx][-1]:02d}_src{src_ids_str}_depth.mp4"
                        if ep is not None:
                            out_name = f"ep{ep}_{out_name}"

                        utils.write_video(os.path.join(
                            data_outdir, out_name), pred_depth_vis)

                        if save_depth_np:
                            out_name_np = out_name[:-3] + 'npy'
                            np.save(os.path.join(
                                data_outdir, out_name_np), cur_depth)

                if 'realestate' in dataname or 'mipnerf360' in dataname or 'ibrnet_llff_test' in dataname:
                    count += 1

        if self.opts.dist:
            self.model_ddp.train()
        else:
            self.model.train()
