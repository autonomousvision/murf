import random
import os
import os.path as osp
import torch
import numpy as np
from glob import glob


import json
from collections import defaultdict
import os.path as osp
from imageio import imread
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from tqdm import tqdm
from scipy.io import loadmat
from PIL import Image

from .transforms import RandomCrop


def augment(rgb, intrinsics, c2w_mat):

    # Horizontal Flip with 50% Probability
    if np.random.uniform(0, 1) < 0.5:
        rgb = rgb[:, ::-1, :]
        tf_flip = np.array([[-1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, 1]])
        c2w_mat = c2w_mat @ tf_flip

    # Crop by aspect ratio
    if np.random.uniform(0, 1) < 0.5:
        py = np.random.randint(1, 32)
        rgb = rgb[py:-py, :, :]
    else:
        py = 0

    if np.random.uniform(0, 1) < 0.5:
        px = np.random.randint(1, 32)
        rgb = rgb[:, px:-px, :]
    else:
        px = 0

    H, W, _ = rgb.shape
    rgb = cv2.resize(rgb, (256, 256))
    xscale = 256 / W
    yscale = 256 / H

    intrinsics[0, 0] = intrinsics[0, 0] * xscale
    intrinsics[1, 1] = intrinsics[1, 1] * yscale

    return rgb, intrinsics, c2w_mat


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
    intrinsics = intrinsics.copy()
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


def parse_pose(pose, timestep):
    timesteps = pose[:, :1]
    timesteps = np.around(timesteps)
    mask = (timesteps == timestep)[:, 0]
    pose_entry = pose[mask][0]
    camera = Camera(pose_entry)

    return camera


def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
              center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def get_camera_pose(scene_path, all_pose_dir, uv, views=1):
    npz_files = sorted(scene_path.glob("*.npz"))
    npz_file = npz_files[0]
    data = np.load(npz_file)
    all_pose_dir = Path(all_pose_dir)

    rgb_files = list(data.keys())

    timestamps = [int(rgb_file.split('.')[0]) for rgb_file in rgb_files]
    sorted_ids = np.argsort(timestamps)

    rgb_files = np.array(rgb_files)[sorted_ids]
    timestamps = np.array(timestamps)[sorted_ids]

    camera_file = all_pose_dir / (str(scene_path.name) + '.txt')
    cam_params = parse_pose_file(camera_file)
    # H, W, _ = data[rgb_files[0]].shape

    # Weird cropping of images
    H, W = 256, 456

    xscale = W / min(H, W)
    yscale = H / min(H, W)

    query = {}
    context = {}

    render_frame = min(128, rgb_files.shape[0])

    query_intrinsics = []
    query_c2w = []
    query_rgbs = []
    for i in range(1, render_frame):
        rgb = data[rgb_files[i]]
        timestep = timestamps[i]

        # rgb = cv2.resize(rgb, (W, H))
        intrinsics = unnormalize_intrinsics(
            cam_params[timestep].intrinsics, H, W)

        intrinsics[0, 2] = intrinsics[0, 2] / xscale
        intrinsics[1, 2] = intrinsics[1, 2] / yscale
        rgb = rgb.astype(np.float32) / 127.5 - 1

        query_intrinsics.append(intrinsics)
        query_c2w.append(cam_params[timestep].c2w_mat)
        query_rgbs.append(rgb)

    context_intrinsics = []
    context_c2w = []
    context_rgbs = []

    if views == 1:
        render_ids = [0]
    elif views == 2:
        render_ids = [0, min(len(rgb_files) - 1, 128)]
    elif views == 3:
        render_ids = [0, min(len(rgb_files) - 1, 128) //
                      2, min(len(rgb_files) - 1, 128)]
    else:
        assert False

    for i in render_ids:
        rgb = data[rgb_files[i]]
        timestep = timestamps[i]
        intrinsics = unnormalize_intrinsics(
            cam_params[timestep].intrinsics, H, W)
        intrinsics[0, 2] = intrinsics[0, 2] / xscale
        intrinsics[1, 2] = intrinsics[1, 2] / yscale

        rgb = rgb.astype(np.float32) / 127.5 - 1

        context_intrinsics.append(intrinsics)
        context_c2w.append(cam_params[timestep].c2w_mat)
        context_rgbs.append(rgb)

    query = {'rgb': torch.Tensor(query_rgbs)[None].float(),
             'cam2world': torch.Tensor(query_c2w)[None].float(),
             'intrinsics': torch.Tensor(query_intrinsics)[None].float(),
             'uv': uv.view(-1, 2)[None, None].expand(1, render_frame - 1, -1, -1)}
    ctxt = {'rgb': torch.Tensor(context_rgbs)[None].float(),
            'cam2world': torch.Tensor(context_c2w)[None].float(),
            'intrinsics': torch.Tensor(context_intrinsics)[None].float()}

    return {'query': query, 'context': ctxt}


class RealEstate10k():
    def __init__(self, img_root, pose_root,
                 num_ctxt_views=2,
                 num_query_views=1,
                 query_sparsity=None,
                 max_num_scenes=None,
                 square_crop=True, augment=True, lpips=False,
                 max_len=-1,
                 random_crop=False,
                 crop_height=224,
                 crop_width=224,
                 **kwargs,
                 ):

        self.num_ctxt_views = num_ctxt_views
        self.num_query_views = num_query_views
        self.query_sparsity = query_sparsity

        self.max_len = max_len

        self.random_crop = random_crop
        if random_crop:
            self.crop_transform = RandomCrop(
                crop_height=crop_height, crop_width=crop_width)

        all_im_dir = Path(img_root)
        self.all_pose_dir = Path(pose_root)
        self.lpips = lpips
        self.eval = eval

        self.all_scenes = sorted(all_im_dir.glob('*/'))
        dummy_img_path = str(next(self.all_scenes[0].glob("*.npz")))

        if max_num_scenes:
            self.all_scenes = list(self.all_scenes)[:max_num_scenes]

        data = np.load(dummy_img_path)
        key = list(data.keys())[0]
        im = data[key]

        H, W = im.shape[:2]
        H, W = 256, 455
        self.H, self.W = H, W
        self.augment = augment

        self.square_crop = square_crop

        xscale = W / min(H, W)
        yscale = H / min(H, W)

        dim = min(H, W)

        self.xscale = xscale
        self.yscale = yscale

        # For now the images are already square cropped
        self.H = 256
        self.W = 455

    def get_name(self):
        dataname = 'realestate'
        return dataname

    def __len__(self):
        return len(self.all_scenes) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        scene_path = self.all_scenes[idx]
        npz_files = sorted(scene_path.glob("*.npz"))

        name = scene_path.name

        pose_file = self.all_pose_dir / (str(scene_path.name) + '.txt')
        if not os.path.exists(pose_file):
            print('no pose:', name)
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))
        pose = parse_pose_file(pose_file)

        if len(npz_files) == 0:
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        npz_file = npz_files[0]
        try:
            data = np.load(npz_file)
        except:
            print(npz_file)
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        rgb_files = list(data.keys())
        window_size = 128

        if len(rgb_files) <= 10:
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        timestamps = [int(rgb_file.split('.')[0]) for rgb_file in rgb_files]
        sorted_ids = np.argsort(timestamps)

        rgb_files = np.array(rgb_files)[sorted_ids]
        timestamps = np.array(timestamps)[sorted_ids]

        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        shift = np.random.randint(low=-1, high=2)

        left_bound = 0
        right_bound = num_frames - 1

        candidate_ids = np.arange(left_bound, right_bound)

        # remove windows between frame -32 to 32
        nframe = 1
        nframe_view = 92

        if len(candidate_ids) < self.num_ctxt_views:
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        id_feats = []

        for i in range(self.num_ctxt_views):
            if len(candidate_ids) == 0:
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

            id_feat = np.random.choice(candidate_ids, size=1,
                                       replace=False)
            candidate_ids = candidate_ids[(candidate_ids < (
                id_feat - nframe_view)) | (candidate_ids > (id_feat + nframe_view))]

            id_feats.append(id_feat.item())

        id_feat = np.array(id_feats)

        if self.num_ctxt_views == 2:
            low = np.min(id_feat) - 64
            high = np.max(id_feat) + 64
            low = max(low, 0)
            high = min(high, num_frames - 1)

            if high <= low:
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

            id_render = np.random.randint(
                low=low, high=high, size=self.num_query_views)
        elif self.num_ctxt_views == 1:
            low = np.min(id_feat) - 64
            high = np.max(id_feat) + 64
            low = max(low, 0)
            high = min(high, num_frames - 1)

            if high <= low:
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

            id_render = np.random.randint(
                low=low, high=high, size=self.num_query_views)
        elif self.num_ctxt_views == 3:
            low = np.min(id_feat) + 64
            high = np.max(id_feat) - 64

            if high <= low:
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

            id_render = np.random.randint(
                low=low, high=high, size=self.num_query_views)
        else:
            assert False

        query_rgbs = []
        query_intrinsics = []
        query_c2w = []

        for id in id_render:
            rgb_file = rgb_files[id]
            rgb = data[rgb_file]

            if rgb.shape[0] == 360:
                rgb = cv2.resize(rgb, (self.W, self.H))

            if self.square_crop:
                rgb = square_crop_img(rgb)

            if rgb.shape[0] != 256 or rgb.shape[1] != 256:
                # print('non 256x256 image')
                return self.__getitem__(0)

            # cam_param = parse_pose(pose, timestamps[id])
            cam_param = pose[timestamps[id]]

            intrinsics = unnormalize_intrinsics(
                cam_param.intrinsics, self.H, self.W)

            if self.square_crop:
                intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

            if self.augment:
                rgb, intrinsics, cam_param.c2w_mat = augment(
                    rgb, intrinsics, cam_param.c2w_mat)

            rgb = rgb.astype(np.float32) / 255.
            img_size = rgb.shape[:2]
            # rgb = rgb.reshape((-1, 3))

            mask_lpips = 0.0

            query_rgbs.append(rgb)
            query_intrinsics.append(intrinsics)
            query_c2w.append(cam_param.c2w_mat)

        ctxt_rgbs = []
        ctxt_intrinsics = []
        ctxt_c2w = []

        for id in id_feat:
            rgb_file = rgb_files[id]
            rgb = data[rgb_file]

            if rgb.shape[0] == 360:
                rgb = cv2.resize(rgb, (self.W, self.H))

            if self.square_crop:
                rgb = square_crop_img(rgb)

            # cam_param = parse_pose(pose, timestamps[id])
            cam_param = pose[timestamps[id]]

            intrinsics = unnormalize_intrinsics(
                cam_param.intrinsics, self.H, self.W)

            if self.square_crop:
                intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

            if self.augment:
                rgb, intrinsics, cam_param.c2w_mat = augment(
                    rgb, intrinsics, cam_param.c2w_mat)

            rgb = rgb.astype(np.float32) / 255.  # [0, 1]

            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(intrinsics)
            ctxt_c2w.append(cam_param.c2w_mat)

        ctxt_rgbs = np.stack(ctxt_rgbs)  # [2, H, W, 3]
        ctxt_intrinsics = np.stack(ctxt_intrinsics)  # [2, 4, 4]
        ctxt_c2w = np.stack(ctxt_c2w)  # [2, 4, 4]

        query_rgbs = np.stack(query_rgbs)
        query_intrinsics = np.stack(query_intrinsics)
        query_c2w = np.stack(query_c2w)

        sample = {}
        sample['images'] = torch.cat((torch.from_numpy(ctxt_rgbs), torch.from_numpy(
            query_rgbs)), dim=0).permute(0, 3, 1, 2).float()  # [V, 3, H, W]
        c2w = torch.cat((torch.from_numpy(ctxt_c2w), torch.from_numpy(
            query_c2w)), dim=0).float()  # [V, 4, 4]
        sample['extrinsics'] = torch.inverse(c2w)  # [V, 4, 4]
        sample['intrinsics'] = np.concatenate(
            (ctxt_intrinsics[:, :3, :3], query_intrinsics[:, :3, :3]), axis=0).astype('float32')  # [V, 3, 3]
        near, far = 1, 100
        sample['near_fars'] = torch.from_numpy(
            np.stack([[near, far]] * (self.num_ctxt_views + 1))).float()  # [V, 2]
        sample['img_wh'] = np.array([256, 256]).astype('int')

        # placeholder
        sample['scene'] = 'none'
        sample['view_ids'] = np.array([0, 0])

        if self.random_crop:
            sample = self.crop_transform(sample)

        return sample


class RealEstate10kTest():
    def __init__(self, img_root, pose_root,
                 num_ctxt_views=2,
                 num_query_views=1, query_sparsity=None,
                 max_num_scenes=None, square_crop=True, augment=False, lpips=False,
                 max_len=-1,
                 fixed_test_set=False,
                 window_size=128,
                 fixed_target_frame=False,  # for visualization purposes
                 **kwargs,
                 ):

        assert augment is False
        self.fixed_test_set = fixed_test_set
        self.window_size = window_size
        self.fixed_target_frame = fixed_target_frame

        self.num_ctxt_views = num_ctxt_views
        self.num_query_views = num_query_views
        self.query_sparsity = query_sparsity

        self.max_len = max_len

        all_im_dir = Path(img_root)
        self.all_pose_dir = Path(pose_root)

        self.lpips = lpips

        self.all_scenes = sorted(all_im_dir.glob('*/'))
        dummy_img_path = str(next(self.all_scenes[0].glob("*.npz")))

        if max_num_scenes:
            self.all_scenes = list(self.all_scenes)[:max_num_scenes]

        data = np.load(dummy_img_path)
        key = list(data.keys())[0]
        im = data[key]

        # print(im.shape)
        H, W = im.shape[:2]
        H, W = 256, 455
        self.H, self.W = H, W
        self.augment = augment

        self.square_crop = square_crop
        # Downsample to be 256 x 256 image
        # self.H, self.W = 256, 455

        xscale = W / min(H, W)
        yscale = H / min(H, W)

        dim = min(H, W)

        self.xscale = xscale
        self.yscale = yscale

        # For now the images are already square cropped
        self.H = 256
        self.W = 455

        # print(f"Resolution is {H}, {W}.")

    def get_name(self):
        dataname = 'realestate_test'
        return dataname

    def __len__(self):
        return len(self.all_scenes) if self.max_len <= 0 else self.max_len

    def __getitem__(self, idx):
        scene_path = self.all_scenes[idx]

        npz_files = sorted(scene_path.glob("*.npz"))

        name = scene_path.name

        pose_file = self.all_pose_dir / (str(scene_path.name) + '.txt')
        if not os.path.exists(pose_file):
            print('no pose:', name)
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))
        pose = parse_pose_file(pose_file)

        if len(npz_files) == 0:
            # print('no npz:', name)
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        npz_file = npz_files[0]
        try:
            data = np.load(npz_file)
        except:
            # print('load npz error:', npz_file)
            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        rgb_files = list(data.keys())

        if len(rgb_files) <= 10:
            if self.fixed_test_set:
                return self.__getitem__(0)
            else:
                # print('rgb < 10:', name)
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        timestamps = [int(rgb_file.split('.')[0]) for rgb_file in rgb_files]
        sorted_ids = np.argsort(timestamps)

        rgb_files = np.array(rgb_files)[sorted_ids]
        timestamps = np.array(timestamps)[sorted_ids]

        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)

        if self.fixed_target_frame:
            target_frame = (num_frames - 1) // 2  # middle as target frame
            start = max(target_frame - self.window_size // 2, 0)
            end = min(target_frame + self.window_size // 2, num_frames - 1)
            mid = target_frame
        else:
            start = 0
            end = min(num_frames - 1, self.window_size)
            mid = end // 2

        if self.num_ctxt_views == 1:
            id_feat = np.array([start])
        elif self.num_ctxt_views == 2:
            id_feat = np.array([start, end])
        elif self.num_ctxt_views == 3:
            id_feat = np.array([start, mid, end])
        else:
            print("More than 3 context views not supported")
            assert False

        id_renders = []

        for i in range(start, end):
            dist = np.abs(id_feat - i).min()

            if self.window_size >= 64:
                if dist > 10:
                    id_renders.append(i)
            else:
                id_renders.append(i)

        if len(id_renders) == 0:
            if self.fixed_test_set:
                return self.__getitem__(0)
            else:
                # print('empty id_renders')
                return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

        if self.fixed_target_frame:
            id_render = target_frame
        elif self.fixed_test_set:
            id_render = id_renders[len(id_renders) // 2]
        else:
            id_render = random.choice(id_renders)

        id_render = np.array([id_render])

        query_rgbs = []
        query_intrinsics = []
        query_c2w = []
        uvs = []

        for id in id_render:
            rgb_file = rgb_files[id]
            rgb = data[rgb_file]

            if rgb.shape[0] == 360:
                rgb = cv2.resize(rgb, (self.W, self.H))

            if self.square_crop:
                rgb = square_crop_img(rgb)

            if rgb.shape[0] != 256 or rgb.shape[1] != 256:
                if self.fixed_test_set:
                    return self.__getitem__(0)
                else:
                    # print('non 256x256 image, skip')
                    return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

            cam_param = pose[timestamps[id]]

            intrinsics = unnormalize_intrinsics(
                cam_param.intrinsics, self.H, self.W)

            if self.square_crop:
                intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

            if self.augment:
                rgb, intrinsics, cam_param.c2w_mat = augment(
                    rgb, intrinsics, cam_param.c2w_mat)

            rgb = rgb.astype(np.float32) / 255.
            img_size = rgb.shape[:2]

            query_rgbs.append(rgb)
            query_intrinsics.append(intrinsics)
            query_c2w.append(cam_param.c2w_mat)

        ctxt_rgbs = []
        ctxt_intrinsics = []
        ctxt_c2w = []

        for id in id_feat:
            rgb_file = rgb_files[id]
            rgb = data[rgb_file]

            if rgb.shape[0] == 360:
                rgb = cv2.resize(rgb, (self.W, self.H))

            if self.square_crop:
                rgb = square_crop_img(rgb)

            cam_param = pose[timestamps[id]]

            intrinsics = unnormalize_intrinsics(
                cam_param.intrinsics, self.H, self.W)

            if self.square_crop:
                intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

            if self.augment:
                rgb, intrinsics, cam_param.c2w_mat = augment(
                    rgb, intrinsics, cam_param.c2w_mat)

            rgb = rgb.astype(np.float32) / 255.

            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(intrinsics)
            ctxt_c2w.append(cam_param.c2w_mat)

        ctxt_rgbs = np.stack(ctxt_rgbs)
        ctxt_intrinsics = np.stack(ctxt_intrinsics)
        ctxt_c2w = np.stack(ctxt_c2w)

        query_rgbs = np.stack(query_rgbs)
        query_intrinsics = np.stack(query_intrinsics)
        query_c2w = np.stack(query_c2w)

        sample = {}
        sample['images'] = torch.cat((torch.from_numpy(ctxt_rgbs), torch.from_numpy(
            query_rgbs)), dim=0).permute(0, 3, 1, 2).float()  # [V, 3, H, W]
        c2w = torch.cat((torch.from_numpy(ctxt_c2w), torch.from_numpy(
            query_c2w)), dim=0).float()  # [V, 4, 4]
        sample['extrinsics'] = torch.inverse(c2w)  # [V, 4, 4]
        sample['intrinsics'] = torch.cat((torch.from_numpy(ctxt_intrinsics)[
                                         :, :3, :3], torch.from_numpy(query_intrinsics)[:, :3, :3]), dim=0).float()  # [V, 3, 3]
        near, far = 1, 100
        sample['near_fars'] = torch.from_numpy(
            np.stack([[near, far]] * (self.num_ctxt_views + 1))).float()  # [V, 2]
        sample['img_wh'] = np.array([256, 256]).astype('int')

        # placeholder
        sample['scene'] = 'none'
        sample['view_ids'] = np.array([0, 0])

        return sample
