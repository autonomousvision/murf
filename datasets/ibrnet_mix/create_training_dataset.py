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


import numpy as np
from . import dataset_dict
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
from typing import Optional
from operator import itemgetter
import torch
import os


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def create_training_dataset(root_dir,
                            n_views=5,
                            train_dataset="llff+spaces+ibrnet_collected+realestate+google_scanned",
                            dataset_weights=[0.3, 0.15, 0.35, 0.15, 0.05],
                            distributed=False,
                            num_replicas=None,
                            rank=None,
                            mixall=False,
                            no_random_view=False,
                            dataset_replicas=None,
                            realestate_full_set=False,
                            realestate_frame_dir=None,
                            mixall_random_view=False,
                            realestate_use_all_scenes=False,
                            ):
    # parse args.train_dataset, "+" indicates that multiple datasets are used, for example "ibrnet_collect+llff+spaces"
    # otherwise only one dataset is used
    # args.dataset_weights should be a list representing the resampling rate for each dataset, and should sum up to 1

    print('training dataset: {}'.format(train_dataset))
    mode = 'train'
    train_dataset_names = train_dataset.split('+')
    weights = dataset_weights

    if dataset_replicas is not None:
        # increase the number of samples per epoch for better training efficiency
        train_dataset_names = train_dataset_names * dataset_replicas
        weights = weights * dataset_replicas
        weights = [x / dataset_replicas for x in weights]

    assert len(train_dataset_names) == len(weights)
    assert np.abs(np.sum(weights) - 1.) < 1e-6
    print(train_dataset_names)
    print('weights:{}'.format(weights))
    train_datasets = []
    train_weights_samples = []

    for training_dataset_name, weight in zip(train_dataset_names, weights):
        if mixall:
            train_dataset = dataset_dict[training_dataset_name](root_dir, split=mode, n_views=n_views,
                                                                no_random_view=not mixall_random_view,
                                                                large_subsample=True,
                                                                include_more_scenes=True,  # ibrnet_collected_more
                                                                full_set=realestate_full_set,
                                                                frame_dir=realestate_frame_dir,
                                                                use_all_scenes=realestate_use_all_scenes,
                                                                )
        else:
            train_dataset = dataset_dict[training_dataset_name](root_dir, split=mode, n_views=n_views,
                                                                no_random_view=no_random_view,
                                                                full_set=realestate_full_set,
                                                                frame_dir=realestate_frame_dir,
                                                                )
        print(training_dataset_name, len(train_dataset))
        train_datasets.append(train_dataset)
        num_samples = len(train_dataset)
        weight_each_sample = weight / num_samples
        train_weights_samples.extend([weight_each_sample]*num_samples)

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_weights = torch.from_numpy(np.array(train_weights_samples))
    sampler = WeightedRandomSampler(train_weights, len(train_weights))
    train_sampler = DistributedSamplerWrapper(
        sampler, num_replicas=num_replicas, rank=rank) if distributed else sampler

    return train_dataset, train_sampler
