#!/usr/bin/env bash


# reproduce the numbers in Table 4 of our paper


# evaluate on llff test set, 4 input views
# 25.95, 0.897, 0.149
CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=4 python test.py \
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
CUDA_VISIBLE_DEVICES=3 python test.py \
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

CHECKPOINT_DIR=checkpoints/tmp && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=2 python test.py \
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











# test on LLFF (final)
# I: 26.49, 0.909, 0.143
CHECKPOINT_DIR=/srv/beegfs02/scratch/neural_rendering/data/checkpoints/murf_ibrnet_fixedviewnumber/baseline-view10-finenerf && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=3 python test.py \
--yaml=test_ibrnet_llff_test \
--load=pretrained/murf-llff-10view-d74cff18.pth \
--no_coarse_init_fine \
--mix_data_train \
--n_src_views=10 \
--no_random_view \
--max_epoch=10 \
--dataset_replicas=4 \
--output_path=${CHECKPOINT_DIR} \
--max_crop_height=256 \
--max_crop_width=384 \
--simple_cond_nerf \
--encoder.use_multiview_gmflow \
--encoder.num_transformer_layers=6 \
--encoder.attn_splits_list=[4] \
--resize_factor=32 \
--with_cnn_features \
--encoder.cos_n_group=[8,8,6,4] \
--weighted_cosine \
--encoder.feature_upsampler=none \
--radiance_subsample_factor=8 \
--sample_color_window_radius=4 \
--multi_view_agg \
--multi_view_agg_new \
--mvagg2 \
--mvfeature_bugfix \
--decoder_input_feature \
--input_viewdir_diff \
--3dresnet_decoder \
--decoder_num_resblocks=12 \
--conv_2plus1d \
--decoder.net_width=128 \
--residual_color_cosine \
--upconv_channel_list=[128,64,16] \
--upconv3d \
--density_softplus_act \
--nerf.sample_intvs=64 \
--train_fine_nerf \
--final_fine_nerf \
--fine_encoder \
--fine_3dconv \
--fine_decoder_type=unet \
--fine_net_width=16 \
--num_fine_samples=16 \
--source_view_include_coarse_pred \
--freeze_coarse_nerf \
--remove_depth_boundary \
--L1_loss \
--loss_weight.ssim=1 \
--loss_weight.lpips=1 \
--data_train.downsample=1. \
--train_no_rand_ray \
--optim.lr_enc=1.e-4 \
--optim.lr_dec=1.e-3 \
--no_val \
--data_train.root_dir=/srv/beegfs02/scratch/neural_rendering/data/datasets/ibrdata \
--data_test.ibrnet_llff_test.root_dir=/srv/beegfs02/scratch/neural_rendering/data/datasets/ibrdata \
--inference_size=[768,1024] \
--inference_splits=4 \
--fine_inference_splits=4 \
--save_imgs



# 10 view model test 6 view
# I: 26.41, 0.905, 0.139 (split 4, final)
# I: 26.42, 0.906, 0.138 (split 2)
CHECKPOINT_DIR=/srv/beegfs02/scratch/neural_rendering/data/checkpoints/murf_ibrnet_fixedviewnumber/baseline-view10-finenerf && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=2 python test.py \
--yaml=test_ibrnet_llff_test \
--load=pretrained/murf-llff-10view-d74cff18.pth \
--no_coarse_init_fine \
--mix_data_train \
--n_src_views=6 \
--no_random_view \
--max_epoch=10 \
--dataset_replicas=4 \
--output_path=${CHECKPOINT_DIR} \
--max_crop_height=256 \
--max_crop_width=384 \
--simple_cond_nerf \
--encoder.use_multiview_gmflow \
--encoder.num_transformer_layers=6 \
--encoder.attn_splits_list=[4] \
--resize_factor=32 \
--with_cnn_features \
--encoder.cos_n_group=[8,8,6,4] \
--weighted_cosine \
--encoder.feature_upsampler=none \
--radiance_subsample_factor=8 \
--sample_color_window_radius=4 \
--multi_view_agg \
--multi_view_agg_new \
--mvagg2 \
--mvfeature_bugfix \
--decoder_input_feature \
--input_viewdir_diff \
--3dresnet_decoder \
--decoder_num_resblocks=12 \
--conv_2plus1d \
--decoder.net_width=128 \
--residual_color_cosine \
--upconv_channel_list=[128,64,16] \
--upconv3d \
--density_softplus_act \
--nerf.sample_intvs=64 \
--train_fine_nerf \
--final_fine_nerf \
--fine_encoder \
--fine_3dconv \
--fine_decoder_type=unet \
--fine_net_width=16 \
--num_fine_samples=16 \
--source_view_include_coarse_pred \
--freeze_coarse_nerf \
--remove_depth_boundary \
--L1_loss \
--loss_weight.ssim=1 \
--loss_weight.lpips=1 \
--data_train.downsample=1. \
--train_no_rand_ray \
--optim.lr_enc=1.e-4 \
--optim.lr_dec=1.e-3 \
--no_val \
--data_train.root_dir=/srv/beegfs02/scratch/neural_rendering/data/datasets/ibrdata \
--data_test.ibrnet_llff_test.root_dir=/srv/beegfs02/scratch/neural_rendering/data/datasets/ibrdata \
--inference_size=[768,1024] \
--inference_splits=4 \
--fine_inference_splits=4



# 10 view model test 3 view
# I: 24.62, 0.875, 0.164,
CHECKPOINT_DIR=/srv/beegfs02/scratch/neural_rendering/data/checkpoints/murf_ibrnet_fixedviewnumber/baseline-view10-finenerf && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=3 python test.py \
--yaml=test_ibrnet_llff_test \
--load=pretrained/murf-llff-10view-d74cff18.pth \
--no_coarse_init_fine \
--mix_data_train \
--n_src_views=3 \
--no_random_view \
--max_epoch=10 \
--dataset_replicas=4 \
--output_path=${CHECKPOINT_DIR} \
--max_crop_height=256 \
--max_crop_width=384 \
--simple_cond_nerf \
--encoder.use_multiview_gmflow \
--encoder.num_transformer_layers=6 \
--encoder.attn_splits_list=[4] \
--resize_factor=32 \
--with_cnn_features \
--encoder.cos_n_group=[8,8,6,4] \
--weighted_cosine \
--encoder.feature_upsampler=none \
--radiance_subsample_factor=8 \
--sample_color_window_radius=4 \
--multi_view_agg \
--multi_view_agg_new \
--mvagg2 \
--mvfeature_bugfix \
--decoder_input_feature \
--input_viewdir_diff \
--3dresnet_decoder \
--decoder_num_resblocks=12 \
--conv_2plus1d \
--decoder.net_width=128 \
--residual_color_cosine \
--upconv_channel_list=[128,64,16] \
--upconv3d \
--density_softplus_act \
--nerf.sample_intvs=64 \
--train_fine_nerf \
--final_fine_nerf \
--fine_encoder \
--fine_3dconv \
--fine_decoder_type=unet \
--fine_net_width=16 \
--num_fine_samples=16 \
--source_view_include_coarse_pred \
--freeze_coarse_nerf \
--remove_depth_boundary \
--L1_loss \
--loss_weight.ssim=1 \
--loss_weight.lpips=1 \
--data_train.downsample=1. \
--train_no_rand_ray \
--optim.lr_enc=1.e-4 \
--optim.lr_dec=1.e-3 \
--no_val \
--data_train.root_dir=/srv/beegfs02/scratch/neural_rendering/data/datasets/ibrdata \
--data_test.ibrnet_llff_test.root_dir=/srv/beegfs02/scratch/neural_rendering/data/datasets/ibrdata \
--inference_size=[768,1024] \
--inference_splits=4 \
--fine_inference_splits=4


# 10 view model test 4 view
# I: 25.55, 0.893, 0.148
CHECKPOINT_DIR=/srv/beegfs02/scratch/neural_rendering/data/checkpoints/murf_ibrnet_fixedviewnumber/baseline-view10-finenerf && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=5 python test.py \
--yaml=test_ibrnet_llff_test \
--load=pretrained/murf-llff-10view-d74cff18.pth \
--no_coarse_init_fine \
--mix_data_train \
--n_src_views=4 \
--no_random_view \
--max_epoch=10 \
--dataset_replicas=4 \
--output_path=${CHECKPOINT_DIR} \
--max_crop_height=256 \
--max_crop_width=384 \
--simple_cond_nerf \
--encoder.use_multiview_gmflow \
--encoder.num_transformer_layers=6 \
--encoder.attn_splits_list=[4] \
--resize_factor=32 \
--with_cnn_features \
--encoder.cos_n_group=[8,8,6,4] \
--weighted_cosine \
--encoder.feature_upsampler=none \
--radiance_subsample_factor=8 \
--sample_color_window_radius=4 \
--multi_view_agg \
--multi_view_agg_new \
--mvagg2 \
--mvfeature_bugfix \
--decoder_input_feature \
--input_viewdir_diff \
--3dresnet_decoder \
--decoder_num_resblocks=12 \
--conv_2plus1d \
--decoder.net_width=128 \
--residual_color_cosine \
--upconv_channel_list=[128,64,16] \
--upconv3d \
--density_softplus_act \
--nerf.sample_intvs=64 \
--train_fine_nerf \
--final_fine_nerf \
--fine_encoder \
--fine_3dconv \
--fine_decoder_type=unet \
--fine_net_width=16 \
--num_fine_samples=16 \
--source_view_include_coarse_pred \
--freeze_coarse_nerf \
--remove_depth_boundary \
--L1_loss \
--loss_weight.ssim=1 \
--loss_weight.lpips=1 \
--data_train.downsample=1. \
--train_no_rand_ray \
--optim.lr_enc=1.e-4 \
--optim.lr_dec=1.e-3 \
--no_val \
--data_train.root_dir=/srv/beegfs02/scratch/neural_rendering/data/datasets/ibrdata \
--data_test.ibrnet_llff_test.root_dir=/srv/beegfs02/scratch/neural_rendering/data/datasets/ibrdata \
--inference_size=[768,1024] \
--inference_splits=4 \
--fine_inference_splits=4

