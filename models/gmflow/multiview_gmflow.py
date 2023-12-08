import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .utils import feature_add_position_list
from .multiview_transformer import MultiViewFeatureTransformer


class MultiViewGMFlow(nn.Module):
    def __init__(self,
                 num_scales=1,
                 with_cnn_features=True,
                 upsample_factor=8,
                 feature_channels=128,
                 attention_type='swin',
                 num_transformer_layers=6,
                 ffn_dim_expansion=4,
                 num_head=1,
                 feature_upsampler='none',
                 add_per_view_attn=False,
                 no_cross_attn=False,
                 **kwargs,
                 ):
        super(MultiViewGMFlow, self).__init__()

        self.num_scales = num_scales
        self.with_cnn_features = with_cnn_features
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_layers = num_transformer_layers
        self.feature_upsampler = feature_upsampler

        # CNN
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=num_scales,
                                   with_cnn_features=with_cnn_features,
                                   )

        # Transformer
        self.transformer = MultiViewFeatureTransformer(num_layers=num_transformer_layers,
                                                       d_model=feature_channels,
                                                       nhead=num_head,
                                                       attention_type=attention_type,
                                                       ffn_dim_expansion=ffn_dim_expansion,
                                                       add_per_view_attn=add_per_view_attn,
                                                       no_cross_attn=no_cross_attn,
                                                       )

    def extract_feature(self, images):
        batch_size, n_img, c, h, w = images.shape
        concat = images.reshape(batch_size*n_img, c, h, w)  # [nB, C, H, W]
        # list of [nB, C, H, W], resolution from high to low
        features = self.backbone(concat)

        if not isinstance(features, list):
            features = [features]

        # reverse: resolution from low to high
        features = features[::-1]

        features_list = [[] for _ in range(n_img)]

        if self.with_cnn_features:
            final_features_list = [[] for _ in range(n_img)]

        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, n_img, 0)  # tuple
            for idx, chunk in enumerate(chunks):
                features_list[idx].append(chunk)

            # only the final cnn features
            if self.with_cnn_features and i == 0:
                for idx, chunk in enumerate(chunks):
                    final_features_list[idx].append(chunk)

        if self.with_cnn_features:
            return features_list, final_features_list

        return features_list

    def normalize_images(self, images):
        '''Normalize image to match the pretrained GMFlow backbone.
            images: (B, N_Views, C, H, W)
        '''
        shape = [*[1]*(images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
            *shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(
            *shape).to(images.device)

        return (images - mean) / std

    def forward(self, images, attn_splits_list=None, **kwargs):
        ''' images: (B, N_Views, C, H, W), range [0, 1] '''
        results_dict = {}
        aug_features_list = []

        # resolution low to high
        features_list = self.extract_feature(
            self.normalize_images(images))  # list of features

        if self.with_cnn_features:
            full_features_list, features_list = features_list

            for scale_idx in range(3):
                cur_features_list = [x[scale_idx] for x in full_features_list]
                ls_feats = torch.stack(
                    cur_features_list, dim=1)  # [B, V, C, H, W]

                aug_features_list.append(ls_feats)

        assert len(attn_splits_list) == self.num_scales

        for scale_idx in range(self.num_scales):
            cur_features_list = [x[scale_idx] for x in features_list]

            attn_splits = attn_splits_list[scale_idx]

            # add position to features
            cur_features_list = feature_add_position_list(
                cur_features_list, attn_splits, self.feature_channels)

            # Transformer
            cur_features_list = self.transformer(
                cur_features_list, attn_num_splits=attn_splits)

            up_features = torch.stack(
                cur_features_list, dim=1)  # [B, V, C, H, W]

            if self.with_cnn_features:
                # 1/8, 1/8, 1/4, 1/2
                aug_features_list.insert(0, up_features)
            else:
                aug_features_list.append(up_features)  # BxVxCxHxW

        results_dict.update({
            'aug_feats_list': aug_features_list
        })

        return results_dict
