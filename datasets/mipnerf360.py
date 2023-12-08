# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
from .ibrnet_mix.data_utils import random_crop, random_flip, get_nearest_pose_ids
from .ibrnet_mix.llff_data_utils import load_llff_data, batch_parse_llff_poses
from torchvision import transforms as T


class MipNeRF360Dataset(Dataset):
    def __init__(self, root_dir, split='test', n_views=3,
                 downsample_factor=8,
                 test_views_method='nearest',
                 view_selection_stride=1,
                 test_scene_name=None,
                 max_len=-1,
                 **kwargs):
        num_source_views = n_views
        llffhold = 8

        self.mode = split  # train / test / validation
        self.num_source_views = num_source_views
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        self.transform = self.define_transforms()
        self.test_views_method = test_views_method
        self.view_selection_stride = view_selection_stride
        self.max_len = max_len

        scenes = sorted(os.listdir(root_dir))
        # some scenes are not available, as .txt files
        scenes = [scene for scene in scenes if os.path.isdir(
            os.path.join(root_dir, scene))]
        # ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump']

        self.test_scene_name = test_scene_name

        if self.test_scene_name is not None:
            scenes = [self.test_scene_name]
            print(scenes[0])

        for i, scene in enumerate(scenes):
            if scene == 'stump':
                continue

            scene_path = os.path.join(root_dir, scene)

            if scene in ['bicycle']:
                _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False, factor=downsample_factor,
                                                                                subsample_factor=2
                                                                                )
            else:
                _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(
                    scene_path, load_imgs=False, factor=downsample_factor)

            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            if self.mode == 'train':
                i_train = np.array(np.arange(int(poses.shape[0])))
                i_render = i_train
            else:
                i_test = np.arange(poses.shape[0])[::llffhold]

                i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                    (j not in i_test and j not in i_test)])
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(
                np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend(
                [intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend(
                [c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend(
                [[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)

    def get_name(self):
        dataname = 'mipnerf360'
        return dataname

    def define_transforms(self):
        transform = T.Compose([T.ToTensor(),])  # (3, h, w)
        return transform

    def __len__(self):
        return len(self.render_rgb_files) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        sample = {}
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        if self.mode == 'train':
            id_render = train_rgb_files.index(rgb_file)
            subsample_factor = np.random.choice(
                np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + \
                np.random.randint(low=-2, high=3)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                min(self.num_source_views *
                                                    subsample_factor, 20),
                                                tar_id=id_render,
                                                angular_dist_method='dist',
                                                view_selection_method=self.test_views_method,
                                                view_selection_stride=self.view_selection_stride,
                                                )
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(
            num_select, len(nearest_pose_ids)), replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(
                len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(
                train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        if self.mode == 'train':
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                             (crop_h, crop_w))

        if self.mode == 'train' and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(
                rgb, camera, src_rgbs, src_cameras)

        depth_range = torch.tensor(
            [depth_range[0] * 0.9, depth_range[1] * 1.6])

        sample['images'] = torch.stack(
            [self.transform(img) for img in [*src_rgbs, rgb]]).float()  # (V, C, H, W)
        # ibrnet camera format: [(h, w, intr(16), extr(16))]
        sample['extrinsics'] = np.stack([np.linalg.inv(
            x[-16:].reshape(4, 4)) for x in [*src_cameras, camera]]).astype(np.float32)  # (V, 4, 4)
        sample['intrinsics'] = np.stack([x[2:-16].reshape(4, 4)[:3, :3]
                                        for x in [*src_cameras, camera]]).astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array([*nearest_pose_ids, id_render])
        sample['scene'] = f"{self.get_name()}_{rgb_file.split('/')[-3]}"
        sample['img_wh'] = np.array([camera[1], camera[0]]).astype('int')
        sample['near_fars'] = np.expand_dims(np.array([depth_range[0].item(), depth_range[1].item(
        )]), axis=0).repeat(sample['view_ids'].shape[0], axis=0).astype(np.float32)

        return sample
