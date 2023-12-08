from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
import cv2

from misc.utils import read_pfm
from .transforms import RandomCrop


class MVSDatasetDTU(Dataset):
    def __init__(self, root_dir, split, n_views=3, img_wh=None, downSample=1.0, max_len=-1,
                 test_views_method='nearest',
                 load_depth=False,
                 eval_depth_5scans=False,
                 random_crop=False,
                 fixed_crop=False,
                 crop_height=384,
                 crop_width=512,
                 train_select_all_views=False,
                 train_nearest=False,
                 test_scan_name=None,  # evaluate on specific scan
                 test_view_stride=None,  # control the baseline
                 continuous_view=False,  # select two consective view from sorted views
                 **kwargs):
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.split = split
        self.n_views = n_views
        self.img_wh = img_wh
        self.downSample = downSample
        self.scale_factor = 1.0 / 200
        self.max_len = max_len
        self.val_light_idx = 3
        self.val_view_idx = 24

        self.load_depth = load_depth
        self.eval_depth_5scans = eval_depth_5scans

        if test_views_method == 'nearest_fixed':
            # random select training views
            train_select_all_views = True

        self.train_select_all_views = train_select_all_views
        self.train_nearest = train_nearest
        self.test_view_stride = test_view_stride
        self.continuous_view = continuous_view

        self.transform = self.define_transforms()

        self.test_scan_name = test_scan_name

        # random crop
        self.random_crop = random_crop
        if random_crop:
            self.crop_transform = RandomCrop(
                crop_height=crop_height, crop_width=crop_width, fixed_crop=fixed_crop)

        if split in ['train', 'val']:
            scene_list_filepath = os.path.join(
                'configs', 'dtu_meta', 'train_all.txt')  # 88 scans
            if self.train_select_all_views:
                id_list = list(range(49))
                self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, self.near_fars_dict = \
                    self.build_camera_info(id_list)
                view_pairs_filepath = os.path.join(
                    'configs', 'dtu_meta', 'view_pairs.txt')
                self.metas, _ = self.build_train_metas(
                    scene_list_filepath, view_pairs_filepath)
            else:
                # 10 src views for each target view, sorted according to scores computed in mvsnet:
                # https://github.com/YoYo000/MVSNet/blob/3ae2cb2b72c6df58ebcb321d7d243d4efd01fbc5/mvsnet/colmap2mvsnet.py#L377
                view_pairs_filepath = os.path.join(
                    'configs', 'dtu_meta', 'view_pairs.txt')
                self.metas, id_list = self.build_train_metas(
                    scene_list_filepath, view_pairs_filepath)
                self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, self.near_fars_dict = \
                    self.build_camera_info(id_list)

        else:  # test cases
            scene_list_filepath = os.path.join(
                'configs', 'dtu_meta', 'val_all.txt')
            view_pairs_filepath = os.path.join('configs', 'pairs.th')
            view_pairs = torch.load(view_pairs_filepath)
            train_views, test_views = view_pairs['dtu_train'], view_pairs['dtu_test']

            if test_views_method in ['nearest_fixed', 'nearest_all_views']:
                # compare with mipnerf, use fixed source view, selected from all the remaining views
                # nearest_all_views: for each test view, sample its nearest view from all remaining views
                train_views = [i for i in range(49) if i not in test_views]

            # train_views: [25, 21, 33, 22, 14, 15, 26, 30, 31, 35, 34, 43, 46, 29, 16, 36]
            # test_views: [32, 24, 23, 44]
            # test samples: 16 scans, each with 4 test views
            id_list = [*train_views, *test_views]
            self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, self.near_fars_dict = \
                self.build_camera_info(id_list)
            self.metas = self.build_test_metas(
                scene_list_filepath, train_views, test_views, method=test_views_method)

    def get_name(self):
        dataname = 'dtu'
        return dataname

    def define_transforms(self):
        transform = T.Compose([T.ToTensor()])
        return transform

    def build_train_metas(self, scene_list_filepath, view_pairs_filepath):
        '''Build train metas, get input source views based on the order pre-defined in `view_pairs_filepath`.'''
        metas = []
        # read scene list
        with open(scene_list_filepath) as f:
            scans = [line.rstrip() for line in f.readlines()]

        # light conditions 0-6 for training
        # light condition 3 for testing
        light_idxs = [
            self.val_light_idx] if 'train' != self.split else range(7)

        id_list = []
        for scan in scans:
            with open(view_pairs_filepath) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())

                    # view selection
                    if self.train_select_all_views:
                        train_views = [x for x in range(
                            num_viewpoint) if x != ref_view]

                        if self.train_nearest:
                            # sort the reference source view accordingly
                            cam_pos_trains = np.stack(
                                [self.cam2worlds_dict[x] for x in train_views])[:, :3, 3]
                            cam_pos_target = self.cam2worlds_dict[ref_view][:3, 3]
                            dis = np.sum(
                                np.abs(cam_pos_trains - cam_pos_target), axis=-1)
                            src_idx = np.argsort(dis)
                            src_views = [train_views[x] for x in src_idx]
                        else:
                            # no need to sort, just random select
                            src_views = train_views

                        # move to next line
                        tmp = [int(x)
                               for x in f.readline().rstrip().split()[1::2]]

                    else:
                        src_views = [int(x)
                                     for x in f.readline().rstrip().split()[1::2]]

                    for light_idx in light_idxs:
                        if self.split == 'val' and ref_view != self.val_view_idx:
                            continue
                        metas += [(scan, light_idx, ref_view, src_views)]
                        id_list.append([ref_view] + src_views)

        id_list = np.unique(id_list)
        return metas, id_list

    def build_camera_info(self, id_list):
        '''Return the camera information for the given id_list'''
        intrinsics, world2cams, cam2worlds, near_fars = {}, {}, {}, {}
        for vid in id_list:
            proj_mat_filename = os.path.join(
                self.root_dir, f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(
                proj_mat_filename)

            intrinsic[:2] *= 4
            intrinsic[:2] = intrinsic[:2] * self.downSample
            intrinsics[vid] = intrinsic

            extrinsic[:3, 3] *= self.scale_factor
            world2cams[vid] = extrinsic
            cam2worlds[vid] = np.linalg.inv(extrinsic)

            near_fars[vid] = near_far

        return intrinsics, world2cams, cam2worlds, near_fars

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsic = np.fromstring(
            ' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsic = extrinsic.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsic = np.fromstring(
            ' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsic = intrinsic.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + \
            float(lines[11].split()[1]) * 192 * self.scale_factor
        near_far = [depth_min, depth_max]
        return intrinsic, extrinsic, near_far

    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[
                           0], dtype=np.float32)  # (1200, 1600)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=self.downSample,
                             fy=self.downSample, interpolation=cv2.INTER_NEAREST)
        return depth_h

    def build_finetune_metas(self, train_views, method='nearest'):
        '''Build test metas, get input source views based on the `method`.'''
        metas = []

        assert self.finetune_scan_name is not None
        scans = [self.finetune_scan_name]
        test_views = train_views

        light_idx = 3
        for scan in scans:
            for target_view in test_views:
                src_views = self.sorted_test_src_views(
                    target_view, train_views, method)
                src_views = src_views[1:]  # exlucde the target view itself

                metas.append((scan, light_idx, target_view, src_views))
        return metas

    def build_test_metas(self, scene_list_filepath, train_views, test_views, method='nearest'):
        '''Build test metas, get input source views based on the `method`.'''
        metas = []
        # read scene list
        with open(scene_list_filepath) as f:
            scans = [line.rstrip() for line in f.readlines()]

        if self.test_scan_name is not None:
            scans = [self.test_scan_name]

        light_idx = 3
        for scan in scans:
            # evaluate 5 scenes for depth metric
            if self.eval_depth_5scans:
                if scan not in ['scan1', 'scan8', 'scan21', 'scan103', 'scan114']:
                    continue

            # use fixed src views for testing
            if method == 'nearest_fixed':
                src_views = self.sorted_test_src_views_fixed(
                    test_views, train_views)
                for target_view in test_views:
                    # all the test views share the same source view
                    metas.append((scan, light_idx, target_view, src_views))
            elif method == 'nearest_all_views':
                for target_view in test_views:
                    src_views = self.sorted_test_src_views(
                        target_view, train_views, 'nearest')
                    metas.append((scan, light_idx, target_view, src_views))

            else:
                for target_view in test_views:
                    src_views = self.sorted_test_src_views(
                        target_view, train_views, method)
                    metas.append((scan, light_idx, target_view, src_views))

        return metas

    def sorted_test_src_views(self, target_view, train_views, method='nearest'):
        if method == "nearest":
            cam_pos_trains = np.stack(
                [self.cam2worlds_dict[x] for x in train_views])[:, :3, 3]
            cam_pos_target = self.cam2worlds_dict[target_view][:3, 3]
            dis = np.sum(np.abs(cam_pos_trains - cam_pos_target), axis=-1)
            src_idx = np.argsort(dis)
            src_idx = [train_views[x] for x in src_idx]
        elif method == "fixed":
            src_idx = train_views
        else:
            raise Exception('Unknown evaluate method [%s]' % method)
        return src_idx

    def sorted_test_src_views_fixed(self, test_views, train_views):
        # use fixed src views for testing, instead of for using different src views for different test views
        cam_pos_trains = np.stack([self.cam2worlds_dict[x] for x in train_views])[
            :, :3, 3]  # [V, 3], V src views
        cam_pos_target = np.stack([self.cam2worlds_dict[x] for x in test_views])[
            :, :3, 3]  # [N, 3], N test views in total
        dis = np.sum(
            np.abs(cam_pos_trains[:, None] - cam_pos_target[None]), axis=(1, 2))
        src_idx = np.argsort(dis)
        src_idx = [train_views[x] for x in src_idx]

        return src_idx

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        sample = {}

        num_views = self.n_views

        scan, light_idx, target_view, src_views = self.metas[idx]

        if self.split == 'train':
            if self.train_select_all_views:
                if self.train_nearest:
                    # random select from V+16 nearest views
                    ids = torch.sort(torch.randperm(
                        min(num_views + 16, len(src_views)))[:num_views])[0]
                    view_ids = [src_views[i] for i in ids] + [target_view]
                else:
                    # random select from all the src_views
                    ids = torch.randperm(len(src_views))[:num_views]
                    view_ids = [src_views[i] for i in ids] + [target_view]
            else:
                max_num_views = 10 if self.split == 'train' else 15  # 16 views in total
                ids = torch.sort(torch.randperm(
                    min(num_views + 2, max_num_views))[:num_views])[0]
                view_ids = [src_views[i] for i in ids] + [target_view]
        else:
            if self.test_view_stride is not None:
                stride = self.test_view_stride
                if self.continuous_view:
                    curr_src_views = src_views[(
                        stride - 1):stride+num_views - 1]
                else:
                    curr_src_views = src_views[(
                        stride - 1)::stride][:num_views]
                    if len(curr_src_views) < num_views:
                        curr_src_views.append(src_views[-1])
                view_ids = curr_src_views + [target_view]
            else:
                view_ids = [src_views[i]
                            for i in range(num_views)] + [target_view]

        # record proj mats between views
        imgs, intrinsics, w2cs, near_fars = [], [], [], []
        depth = None  # only used for test case
        img_wh = np.round(np.array(self.img_wh) *
                          self.downSample).astype('int')
        for vid in view_ids:
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,
                                        f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')

            img = Image.open(img_filename)
            img = img.resize(img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs.append(img)

            intrinsics.append(self.intrinsics_dict[vid])
            w2cs.append(self.world2cams_dict[vid])
            near_fars.append(self.near_fars_dict[vid])

            # read target view depth for evaluation
            if (self.split in ['test', 'val'] and vid == target_view) or self.load_depth:
                depth_filename = os.path.join(
                    self.root_dir, f'Depths/{scan}/depth_map_{vid:04d}.pfm')
                assert os.path.exists(
                    depth_filename), "Must provide depth for evaluating purpose."
                depth = self.read_depth(depth_filename)
                depth = depth * self.scale_factor

        sample['images'] = torch.stack(imgs).float()  # (V, 3, H, W)
        sample['extrinsics'] = np.stack(w2cs).astype(np.float32)  # (V, 4, 4)
        sample['intrinsics'] = np.stack(
            intrinsics).astype(np.float32)  # (V, 3, 3)
        sample['near_fars'] = np.stack(near_fars).astype(np.float32)
        sample['view_ids'] = np.array(view_ids)
        sample['scene'] = scan
        sample['img_wh'] = img_wh

        if (self.split in ['test', 'val'] and depth is not None) or self.load_depth:
            sample['depth'] = depth.astype(np.float32)

        if 'depth' in sample:
            sample['depth_gt'] = sample['depth']

        if self.random_crop:
            sample = self.crop_transform(sample)

        return sample
