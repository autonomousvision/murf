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
import glob
import cv2
from torchvision import transforms as T


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics


def parse_pose_file(file):
    f = open(file, 'r')
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        entry = [float(x) for x in line.split()]
        id = int(entry[0])
        cam_params[id] = Camera(entry)
    return cam_params


class RealEstateDataset(Dataset):
    def __init__(self, root_dir, split='train', n_views=3,
                 no_random_view=False,
                 large_subsample=False,
                 full_set=False,
                 frame_dir=None,
                 use_all_scenes=False,
                 **kwargs):
        self.use_all_scenes = use_all_scenes

        if full_set:
            self.folder_path = os.path.join(os.path.dirname(
                root_dir), 'RealEstate10KFull720p/data/')
        else:
            self.folder_path = os.path.join(root_dir, 'RealEstate10K-subset/')

        self.mode = split  # train / test / validation
        self.num_source_views = n_views
        self.target_h, self.target_w = 450, 800
        assert split in ['train'], "real estate only for training"
        self.frame_dir = frame_dir if frame_dir is not None else 'frames'
        if use_all_scenes:
            self.scene_path_list = glob.glob(os.path.join(
                self.folder_path, 'frames_camera_*', split, '*'))
        elif full_set:
            self.scene_path_list = glob.glob(os.path.join(
                self.folder_path, self.frame_dir, split, '*'))
        else:
            self.scene_path_list = glob.glob(os.path.join(
                self.folder_path, split, self.frame_dir, '*'))
        self.transform = self.define_transforms()

        self.no_random_view = no_random_view
        self.large_subsample = large_subsample

        all_rgb_files = []
        all_timestamps = []
        for i, scene_path in enumerate(self.scene_path_list):
            rgb_files = [os.path.join(scene_path, f)
                         for f in sorted(os.listdir(scene_path))]
            if len(rgb_files) < 10:
                # print('omitting {}, too few images'.format(os.path.basename(scene_path)))
                continue
            timestamps = [int(os.path.basename(rgb_file).split('.')[0])
                          for rgb_file in rgb_files]
            sorted_ids = np.argsort(timestamps)
            all_rgb_files.append(np.array(rgb_files)[sorted_ids])
            all_timestamps.append(np.array(timestamps)[sorted_ids])

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_timestamps = np.array(all_timestamps)[index]

    def get_name(self):
        dataname = 'ibrnet_realestate'
        return dataname

    def define_transforms(self):
        transform = T.Compose([T.ToTensor()])  # (3, h, w)
        return transform

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        sample = {}
        rgb_files = self.all_rgb_files[idx]
        timestamps = self.all_timestamps[idx]

        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)

        if self.large_subsample:
            window_size = np.random.randint(32, 96)
        else:
            window_size = 32
        shift = np.random.randint(low=-1, high=2)
        id_render = np.random.randint(low=4, high=num_frames-4-1)

        right_bound = min(id_render + window_size + shift, num_frames-1)
        left_bound = max(0, right_bound - 2 * window_size)
        candidate_ids = np.arange(left_bound, right_bound)
        # remove the query frame itself with high probability
        if np.random.choice([0, 1], p=[0.01, 0.99]):
            candidate_ids = candidate_ids[candidate_ids != id_render]

        id_feat = np.random.choice(candidate_ids, size=min(self.num_source_views, len(candidate_ids)),
                                   replace=False)

        if len(id_feat) < self.num_source_views:
            return self.__getitem__(np.random.randint(0, len(self.all_rgb_files)))

        rgb_file = rgb_files[id_render]
        rgb = imageio.imread(rgb_files[id_render])
        # resize the image to target size
        rgb = cv2.resize(rgb, dsize=(self.target_w, self.target_h),
                         interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.

        if self.use_all_scenes:
            camera_file = os.path.dirname(rgb_file).replace(os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(rgb_file)))), 'cameras') + '.txt'
        else:
            camera_file = os.path.dirname(rgb_file).replace(
                self.frame_dir, 'cameras') + '.txt'
        cam_params = parse_pose_file(camera_file)
        cam_param = cam_params[timestamps[id_render]]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size),
                                 unnormalize_intrinsics(
                                     cam_param.intrinsics, self.target_h, self.target_w).flatten(),
                                 cam_param.c2w_mat.flatten())).astype(np.float32)

        depth_range = torch.tensor([1., 100.])

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id])
            # resize the image to target size
            src_rgb = cv2.resize(src_rgb, dsize=(
                self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            src_rgb = src_rgb.astype(np.float32) / 255.
            src_rgbs.append(src_rgb)

            img_size = src_rgb.shape[:2]
            cam_param = cam_params[timestamps[id]]
            src_camera = np.concatenate((list(img_size),
                                         unnormalize_intrinsics(cam_param.intrinsics,
                                                                self.target_h, self.target_w).flatten(),
                                         cam_param.c2w_mat.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        sample['images'] = torch.stack(
            [self.transform(img) for img in [*src_rgbs, rgb]]).float()  # (V, C, H, W)
        # ibrnet camera format: [(h, w, intr(16), extr(16))]
        sample['extrinsics'] = np.stack([np.linalg.inv(
            x[-16:].reshape(4, 4)) for x in [*src_cameras, camera]]).astype(np.float32)  # (V, 4, 4)
        sample['intrinsics'] = np.stack([x[2:-16].reshape(4, 4)[:3, :3]
                                        for x in [*src_cameras, camera]]).astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array([*id_feat, id_render])
        sample['scene'] = f"{self.get_name()}_{rgb_file.split('/')[-2]}"
        sample['img_wh'] = np.array([camera[1], camera[0]]).astype('int')
        sample['near_fars'] = np.expand_dims(np.array([depth_range[0].item(), depth_range[1].item(
        )]), axis=0).repeat(sample['view_ids'].shape[0], axis=0).astype(np.float32)

        return sample
