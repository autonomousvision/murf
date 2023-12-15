#!/usr/bin/env bash


# reproduce the numbers in Table 5 of our paper


# evaluate on mipnerf360 test set, 2 input views, with the model trained realestate10k
# 23.98, 0.800, 0.293
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-realestate10k-2view-74b3217d.pth \
--yaml=test_mipnerf360 \
--n_src_views=2 \
--radiance_subsample_factor=4 \
--sample_color_window_radius=2 \
--decoder.net_width=64 \
--upconv_channel_list=[64,16] \
--data_test.mipnerf360.root_dir=UPDATE_WITH_YOUR_DATA_PATH


# evaluate on mipnerf360 test set, 2 input views, with the model further finetuned on the mixed datasets
# 25.30, 0.850, 0.192
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-mipnerf360-2view-42df3b73.pth \
--yaml=test_mipnerf360 \
--n_src_views=2 \
--radiance_subsample_factor=4 \
--sample_color_window_radius=2 \
--decoder.net_width=64 \
--upconv_channel_list=[64,16] \
--with_fine_nerf \
--fine_inference_splits=2 \
--data_test.mipnerf360.root_dir=UPDATE_WITH_YOUR_DATA_PATH

