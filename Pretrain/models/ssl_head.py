# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
import torch.nn as nn

from .swin_uneter import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
from monai import transforms
from einops import rearrange

transform = transforms.Compose(
    [
        transforms.RandAdjustContrast(prob=0.7, gamma=[0.7, 1.5]),
        transforms.RandShiftIntensity(0.4, prob=0.4),
        transforms.RandGaussianNoise(prob=0.5, mean=0.05, std=0.05),
    ]
)


def aug_rand(samples):
    samples = transform(samples)
    return samples


class SSLHead(nn.Module):
    def __init__(self, args, upsample="vae", dim=768):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            dynamic_masking=args.dynamic_masking,
            hierarchical_masking=args.hierarchical_masking,
            basic_mask_ratio=args.basic_mask_ratio,
            scale=args.scale,
            drop_ratio=args.drop_ratio
        )

        self.swinViT_contrastive = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            dynamic_masking=args.dynamic_masking,
            hierarchical_masking=args.hierarchical_masking,
            basic_mask_ratio=args.basic_mask_ratio,
            scale=args.scale,
            drop_ratio=args.drop_ratio,
            mask_layer=False,
        )

        self.contrastive_pre = nn.Identity()

        self.contrastive_head_x = nn.Linear(dim, 512)
        self.contrastive_head_y = nn.Linear(dim, 512)

        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args.in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 2),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 4),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 8),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm3d(dim // 16),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(dim // 16, args.in_channels, kernel_size=1, stride=1),
            )

    def copy_weight(self):
        for (_, ema_param), (_, model_param) in zip(self.swinViT_contrastive.named_parameters(), self.swinViT.named_parameters()):
            ema_param.data = model_param.data
            ema_param.requires_grad = False

    def forward(self, x, y):
        self.copy_weight()
        y = aug_rand(y)

        x_out = self.swinViT(x.contiguous())[4]
        with torch.no_grad():
            y_out = self.swinViT_contrastive(y.contiguous())[4]

        _, c, h, w, d = x_out.shape

        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        # x4_reshape = rearrange(x4_reshape, 'b n c -> (b n) c')

        x_contrastive = self.contrastive_pre(x4_reshape)
        x_contrastive = self.contrastive_head_x(x_contrastive)

        y_reshape = y_out.flatten(start_dim=2, end_dim=4)
        y_reshape = y_reshape.transpose(1, 2)
        # y_reshape = rearrange(y_reshape, 'b n c -> (b n) c')
        y_contrastive = self.contrastive_pre(y_reshape)
        y_contrastive = self.contrastive_head_y(y_contrastive)

        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        x_rec = self.conv(x_rec)
        return x_contrastive, y_contrastive,  x_rec




