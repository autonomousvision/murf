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


from torchvision import transforms as T
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids
import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
sys.path.append('../')


class GoogleScannedDataset(Dataset):
    def __init__(self, root_dir, split='train', n_views=3, **kwargs):
        self.folder_path = os.path.join(root_dir, 'google_scanned_objects/')
        self.num_source_views = n_views
        self.rectify_inplane_rotation = False
        self.scene_path_list = glob.glob(os.path.join(self.folder_path, '*'))
        self.transform = self.define_transforms()

        all_rgb_files = []
        all_pose_files = []
        all_intrinsics_files = []
        num_files = 250
        for i, scene_path in enumerate(self.scene_path_list):
            rgb_files = [os.path.join(scene_path, 'rgb', f)
                         for f in sorted(os.listdir(os.path.join(scene_path, 'rgb')))]
            pose_files = [f.replace('rgb', 'pose').replace(
                'png', 'txt') for f in rgb_files]
            intrinsics_files = [f.replace('rgb', 'intrinsics').replace(
                'png', 'txt') for f in rgb_files]

            if np.min([len(rgb_files), len(pose_files), len(intrinsics_files)]) \
                    < num_files:
                print(scene_path)
                continue

            all_rgb_files.append(rgb_files)
            all_pose_files.append(pose_files)
            all_intrinsics_files.append(intrinsics_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_pose_files = np.array(all_pose_files)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files)[index]

    def get_name(self):
        dataname = 'google_scanned'
        return dataname

    def define_transforms(self):
        transform = T.Compose([T.ToTensor(),])  # (3, h, w)
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        return transform

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        sample = {}
        rgb_files = self.all_rgb_files[idx]

        pose_files = self.all_pose_files[idx]
        intrinsics_files = self.all_intrinsics_files[idx]

        id_render = np.random.choice(np.arange(len(rgb_files)))
        train_poses = np.stack([np.loadtxt(file).reshape(4, 4)
                               for file in pose_files], axis=0)
        render_pose = train_poses[id_render]
        subsample_factor = np.random.choice(
            np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

        id_feat_pool = get_nearest_pose_ids(render_pose,
                                            train_poses,
                                            self.num_source_views*subsample_factor,
                                            tar_id=id_render,
                                            angular_dist_method='vector')
        id_feat = np.random.choice(
            id_feat_pool, self.num_source_views, replace=False)

        assert id_render not in id_feat
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]):
            id_feat[np.random.choice(len(id_feat))] = id_render

        rgb = imageio.imread(rgb_files[id_render]).astype(np.float32) / 255.

        render_intrinsics = np.loadtxt(intrinsics_files[id_render])
        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), render_intrinsics, render_pose.flatten())).astype(np.float32)

        # get depth range
        min_ratio = 0.1
        origin_depth = np.linalg.inv(render_pose)[2, 3]
        max_radius = 0.5 * np.sqrt(2) * 1.1
        near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        far_depth = origin_depth + max_radius

        src_rgbs = []
        src_intrs = []
        src_poses = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.
            pose = np.loadtxt(pose_files[id])
            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(
                    pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            intrinsics = np.loadtxt(intrinsics_files[id])
            src_intrs.append(intrinsics)
            src_poses.append(pose)

        sample['images'] = torch.stack(
            [self.transform(img) for img in [*src_rgbs, rgb]]).float()  # (V, C, H, W)
        sample['extrinsics'] = np.stack([np.linalg.inv(x.reshape(4, 4)) for x in [
                                        *src_poses, render_pose]]).astype(np.float32)  # (V, 4, 4)
        sample['intrinsics'] = np.stack([x.reshape(4, 4)[:3, :3] for x in [
                                        *src_intrs, render_intrinsics]]).astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array([*id_feat, id_render])
        sample['scene'] = f"{self.get_name()}_{rgb_files[0].split('/')[-3]}"
        sample['img_wh'] = np.array([img_size[1], img_size[0]]).astype('int')
        sample['near_fars'] = np.expand_dims(np.array([near_depth, far_depth]), axis=0).repeat(
            sample['view_ids'].shape[0], axis=0).astype(np.float32)

        return sample
