from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
import cv2

from misc.utils import read_pfm
from .transforms import RandomCrop


class MVSDatasetDTURegNeRF(Dataset):
    """
    RegNeRF setting: https://github.com/google-research/google-research/blob/master/regnerf/internal/datasets.py#L1228
    """

    def __init__(self, root_dir, split, n_views=3, img_wh=None, downSample=1.0, max_len=-1,
                 test_views_method='fixed',
                 random_crop=False,
                 crop_height=384,
                 crop_width=512,
                 min_views=None,
                 max_views=None,
                 render_scan_id=None,  # render a specific scan
                 **kwargs):

        assert split in ['train', 'val', 'test']

        downSample = 1.0  # use the img_wh parameter

        self.root_dir = root_dir
        self.split = split
        self.n_views = n_views
        self.img_wh = img_wh
        self.downSample = downSample
        self.scale_factor = 1.0 / 200
        self.max_len = max_len
        self.val_light_idx = 3
        self.val_view_idx = 24

        # random select number of views for training
        self.min_views = min_views
        self.max_views = max_views

        self.render_scan_id = render_scan_id

        self.transform = self.define_transforms()

        # random crop
        self.random_crop = random_crop
        if random_crop:
            self.crop_transform = RandomCrop(
                crop_height=crop_height, crop_width=crop_width)

        if split in ['train', 'val']:
            scene_list_filepath = os.path.join(
                'configs', 'dtu_meta', 'regnerf_train.txt')  # 109 scans

            # use all available views for training
            id_list = list(range(49))
            self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, self.near_fars_dict = \
                self.build_camera_info(id_list)
            view_pairs_filepath = os.path.join(
                'configs', 'dtu_meta', 'view_pairs.txt')  # not used
            self.metas, _ = self.build_train_metas(
                scene_list_filepath, view_pairs_filepath)

        else:  # test cases
            scene_list_filepath = os.path.join(
                'configs', 'dtu_meta', 'regnerf_test.txt')
            # https://github.com/google-research/google-research/blob/master/regnerf/internal/datasets.py#L1301-L1303
            # 9 train views in regnerf
            train_views = [25, 22, 28, 40, 44, 48, 0, 8, 13]
            # 15 test scenes, each with 25 test views
            test_views = [1, 2, 9, 10, 11, 12, 14, 15, 23, 24, 26,
                          27, 29, 30, 31, 32, 33, 34, 35, 41, 42, 43, 45, 46, 47]
            id_list = [*train_views, *test_views]
            self.intrinsics_dict, self.world2cams_dict, self.cam2worlds_dict, self.near_fars_dict = \
                self.build_camera_info(id_list)
            self.metas = self.build_test_metas(
                scene_list_filepath, train_views, test_views, method=test_views_method)

    def get_name(self):
        dataname = 'dtu_regnerf'
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
                    train_views = [x for x in range(
                        num_viewpoint) if x != ref_view]

                    # sort the reference source view accordingly
                    if self.train_random_nearest:
                        cam_pos_trains = np.stack(
                            [self.cam2worlds_dict[x] for x in train_views])[:, :3, 3]
                        cam_pos_target = self.cam2worlds_dict[ref_view][:3, 3]
                        dis = np.sum(
                            np.abs(cam_pos_trains - cam_pos_target), axis=-1)
                        src_idx = np.argsort(dis)
                        src_views = [train_views[x] for x in src_idx]
                    else:
                        # no need to sort, just random sample
                        src_views = train_views

                    # move to next line
                    tmp = [int(x) for x in f.readline().rstrip().split()[1::2]]

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
                self.root_dir, f'Calibration/cal18/pos_{(vid + 1):03d}.txt')

            with open(proj_mat_filename, 'rb') as f:
                projection = np.loadtxt(f, dtype=np.float32)

            # Decompose projection matrix into pose and camera matrix.
            camera_mat, rot_mat, t = cv2.decomposeProjectionMatrix(projection)[
                :3]
            camera_mat = camera_mat / camera_mat[2, 2]
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = rot_mat.transpose()
            pose[:3, 3] = (t[:3] / t[3])[:, 0]

            extrinsic = np.linalg.inv(pose)

            pose = pose[:3]

            intrinsic = camera_mat

            near_far = [2.125, 4.525]

            # scale intrinsics
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

    def read_mask(self, filename):
        # object mask for evaluation
        image = np.array(Image.open(filename), dtype=np.float32)[
            :, :, :3] / 255.
        image = (image == 1).astype(np.float32)  # [H, W, 3]

        # same model INTER_NEAREST as regnerf
        image = cv2.resize(image, self.img_wh, cv2.INTER_NEAREST)

        return image[:, :, 0]  # [H, W]

    def build_test_metas(self, scene_list_filepath, train_views, test_views, method='nearest'):
        '''Build test metas, get input source views based on the `method`.'''
        metas = []
        # read scene list
        with open(scene_list_filepath) as f:
            scans = [line.rstrip() for line in f.readlines()]

        if self.render_scan_id is not None:
            scans = [self.render_scan_id]

        light_idx = 3
        for scan in scans:
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

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        sample = {}

        if self.split == 'train' and self.min_views is not None and self.max_views is not None:
            num_views = np.random.randint(self.min_views, self.max_views + 1)
        else:
            num_views = self.n_views

        scan, light_idx, target_view, src_views = self.metas[idx]

        if self.split == 'train':
            # random select from all the src_views
            ids = torch.randperm(len(src_views))[:num_views]

            view_ids = [src_views[i] for i in ids] + [target_view]
        else:
            view_ids = [src_views[i] for i in range(num_views)] + [target_view]

        # record proj mats between views
        imgs, intrinsics, w2cs, near_fars = [], [], [], []
        depth = None  # only used for test case
        mask = None  # only used for evaluation
        img_wh = np.array(self.img_wh)
        for vid in view_ids:
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,
                                        f'Rectified/{scan}/rect_{vid + 1:03d}_{light_idx}_r5000.png')

            img = Image.open(img_filename)

            raw_img_wh = img.size

            # same mode cv2.INTER_AREA as regnerf
            img = cv2.resize(np.array(img), img_wh, cv2.INTER_AREA)

            img = self.transform(img)
            imgs.append(img)

            # rescale camera intrinsic
            rescale_factor_x = img_wh[0] / raw_img_wh[0]
            rescale_factor_y = img_wh[1] / raw_img_wh[1]

            cur_intr = self.intrinsics_dict[vid].copy()
            cur_intr[:1] = cur_intr[:1] * rescale_factor_x
            cur_intr[1:2] = cur_intr[1:2] * rescale_factor_y

            intrinsics.append(cur_intr)
            w2cs.append(self.world2cams_dict[vid])
            near_fars.append(self.near_fars_dict[vid])

        sample['images'] = torch.stack(imgs).float()  # (V, 3, H, W)
        sample['extrinsics'] = np.stack(w2cs).astype(np.float32)  # (V, 4, 4)
        sample['intrinsics'] = np.stack(
            intrinsics).astype(np.float32)  # (V, 3, 3)
        sample['near_fars'] = np.stack(near_fars).astype(np.float32)
        sample['view_ids'] = np.array(view_ids)
        sample['scene'] = scan
        sample['img_wh'] = img_wh

        if self.split == 'test':
            idr_scans = ['scan40', 'scan55', 'scan63', 'scan110', 'scan114']
            mask_path = os.path.join(self.root_dir, 'idrmasks')
            if scan in idr_scans:
                def maskf_fn(x): return os.path.join(
                    mask_path, scan, 'mask', f'{x:03d}.png')
            else:
                def maskf_fn(x): return os.path.join(
                    mask_path, scan, f'{x:03d}.png')

            mask_filename = maskf_fn(target_view)
            mask = self.read_mask(mask_filename)

            sample['mask'] = mask  # [H, W]

        if self.random_crop:
            sample = self.crop_transform(sample)

        return sample
