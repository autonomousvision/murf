#!/usr/bin/env bash


# reproduce the numbers in Table 4 of our paper


# evaluate on llff test set, 4 input views
# 25.95, 0.897, 0.149
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python test.py \
--output_path=${CHECKPOINT_DIR} \
--yaml=test_ibrnet_llff_test \
--load=pretrained/murf-llff-6view-15d3646e.pth \
--n_src_views=4 \
--encoder.attn_splits_list=[4] \
--resize_factor=32 \
--weighted_cosine \
--with_fine_nerf \
--data_test.ibrnet_llff_test.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
--inference_size=[768,1024] \
--inference_splits=4 \
--fine_inference_splits=4


# evaluate on llff test set, 6 input views
# 26.04, 0.900, 0.153
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python test.py \
--output_path=${CHECKPOINT_DIR} \
--yaml=test_ibrnet_llff_test \
--load=pretrained/murf-llff-6view-15d3646e.pth \
--n_src_views=6 \
--encoder.attn_splits_list=[4] \
--resize_factor=32 \
--weighted_cosine \
--with_fine_nerf \
--data_test.ibrnet_llff_test.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
--inference_size=[768,1024] \
--inference_splits=4 \
--fine_inference_splits=4


# evaluate on llff test set, 10 input views
# 26.49, 0.909, 0.143
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0 python test.py \
--output_path=${CHECKPOINT_DIR} \
--yaml=test_ibrnet_llff_test \
--load=pretrained/murf-llff-10view-d74cff18.pth \
--n_src_views=10 \
--encoder.attn_splits_list=[4] \
--resize_factor=32 \
--weighted_cosine \
--with_fine_nerf \
--data_test.ibrnet_llff_test.root_dir=UPDATE_WITH_YOUR_DATA_PATH \
--inference_size=[768,1024] \
--inference_splits=4 \
--fine_inference_splits=4
