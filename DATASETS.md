# Datasets

## DTU (small baseline)

* Download the preprocessed DTU training data [dtu_training.rar](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view). Also download [Depth_raw.zip](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip) if would like to evaluate the depth accuracy, otherwise no depth is needed for training.

* Extract `Cameras/` and `Rectified/` from the above downloaded `dtu_training.rar`, and optionally extract `Depths` from the `Depth_raw.zip`. Link the folders to `DTU`, which should have the following structure:

```bash
DTU
├── Cameras
└── Rectified
```



## DTU (large baseline)

- Please refer to [RegNeRF](https://github.com/google-research/google-research/tree/master/regnerf#dtu-dataset) for the downloading of the DTU dataset.
- The folder structure:

```bash
DTURaw/
├── Calibration
├── idrmasks
└── Rectified
```



## RealEstate10K

- Please refer to [AttnRend](https://github.com/yilundu/cross_attention_renderer/tree/master/data_download) for the downloading of the RealEstate10K dataset.
- The folder structure (`data_download` contains video frames, and `RealEstate10K` contains camera poses):

```bash
realestate_full
├── data_download
│   ├── test
│   └── train
└── RealEstate10K
    ├── test
    └── train
```

- The full RealEstate10K dataset is very large, which can be challenging to download. We use a [subset](https://www.dropbox.com/s/qo8b7odsms722kq/cvpr2023_wide_baseline_data.tar.gz?dl=0) provided by [AttnRend](https://github.com/yilundu/cross_attention_renderer#get-started) for ablation experiments in our paper.
- The folder structure of the subset:

```bash
realestate_subset
├── data_download
│   └── realestate
│       ├── test
│       └── train
└── poses
    └── realestate
        ├── test.mat
        └── train.mat
```



## LLFF

- Please refer to [IBRNet](https://github.com/googleinterns/IBRNet#1-training-datasets) for the downloading of the mixed training datasets.
- Download the LLFF test data with:

```bash
gdown https://drive.google.com/uc?id=16VnMcF1KJYxN9QId6TClMsZRahHNMW5g
unzip nerf_llff_data.zip
```

- The folder structure:

```bash
mixdata1/
├── google_scanned_objects
├── ibrnet_collected_1
├── ibrnet_collected_2
├── nerf_llff_data
├── nerf_synthetic
├── RealEstate10K-subset
├── real_iconic_noface
└── spaces_dataset
```



## Mip-NeRF 360 dataset

- Download the dataset with

```
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
```

- The folder structure:

```bash
mipnerf360/
├── bicycle
├── bonsai
├── counter
├── flowers.txt
├── garden
├── kitchen
├── room
├── stump
└── treehill.txt
```







