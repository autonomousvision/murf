#!/usr/bin/env bash


# NOTE: to render videos, you should have `ffmpeg` installed


# render videos from 2 input views
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=7 python test.py \
--output_path=${CHECKPOINT_DIR} \
--load=pretrained/murf-mipnerf360-2view-42df3b73.pth \
--yaml=test_video_mipnerf360 \
--n_src_views=2 \
--radiance_subsample_factor=4 \
--sample_color_window_radius=2 \
--decoder.net_width=64 \
--upconv_channel_list=[64,16] \
--with_fine_nerf \
--fine_inference_splits=2 \
--data_test.mipnerf360.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
--no_resize_back


