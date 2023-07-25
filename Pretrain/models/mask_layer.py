import monai.transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from skimage.measure import shannon_entropy
from einops import rearrange

import SimpleITK as sitk
from time import time


class GridMaskLayer(nn.Module):
    """
    对输入图像进行动态遮挡
    """
    def __init__(self, dynamic_masking=True, hierarchical_masking=None, basic_mask_ratio=0.5,
                 scale=0.5, drop_ratio=0.3):
        super(GridMaskLayer, self).__init__()
        self.pre_defined_grid_resolution = [4, 8, 16]

        # mask ratio 缩放尺度
        self.scale = scale
        self.drop_ratio = drop_ratio
        self.step = 0

        self.dynamic_masking = dynamic_masking
        self.hierarchical_masking = None if hierarchical_masking == 0. else hierarchical_masking
        self.basic_mask_ratio = basic_mask_ratio

    def forward(self, x):
        if self.step == 0:
            x = self.mask_token(x)
        else:
            if np.random.random(1) < self.drop_ratio and self.hierarchical_masking is not None:
                x = self.mask_token(x)
            else:
                x = x

        self.step += 1
        self.update_next_step_mask_ratio()
        return x

    def initial(self, x):
        self.step = 0
        # [b]
        self.grid_resolution = []
        # [b, grid_num]
        self.shannon_entropy = []
        # [b, grid_num]
        self.dynamic_mask_ratio = []
        self.mask_list = []

        self.get_grid_resolution(x)
        self.get_dynamic_mask_ratio()

    def get_grid_resolution(self, x):
        b, c, z, h, w = x.size()

        for batch_index in range(b):
            image = x[batch_index]

            entropy_list = []
            std_list = []
            for grid in self.pre_defined_grid_resolution:
                # Apply grid
                image = image.contiguous()
                image = image.view(c, grid, z // grid, grid, h // grid, grid, w // grid)
                grid_num = grid * grid * grid
                # token_num_in_grid = tsz * tsh * tsw // grid_num
                image = image.permute(0, 1, 3, 5, 2, 4, 6)
                # grid: [grid_num, token_num_in_grid, token_size]
                image = rearrange(image, 'c res_z res_s res_w z h w -> c (res_z res_s res_w) (z h w)')
                image = image.permute(1, 0, 2)

                # Get shannon_entropy
                entropy = []
                for grid_index in range(grid_num):
                    entropy.append(shannon_entropy(image[grid_index]))
                entropy_list.append(entropy)
                # Get the standard deviation of the Shannon entropy under the grid resolution
                std_list.append(np.power(grid, -0.1) * np.std(entropy) / np.mean(entropy))

            # Store the grid_resolution of # image in the batch
            self.grid_resolution.append(self.pre_defined_grid_resolution[np.argmax(std_list)])
            self.shannon_entropy.append(entropy_list[np.argmax(std_list)])

    def get_dynamic_mask_ratio(self):
        # mask_ratio_i = base_mask_ratio * (1 + index * sample[-self.scale, self.scale] / n )
        if self.dynamic_masking is True:
            for batch_index in range(len(self.shannon_entropy)):
                entropy_sorted = sorted(list(set(self.shannon_entropy[batch_index])))

                ratio_list = np.ones_like(np.asarray(self.shannon_entropy[batch_index], dtype=np.float_))

                mask_ratio_sup = 1 + self.scale
                mask_ratio_inf = 1 - self.scale
                mask_ratio_linear = np.linspace(mask_ratio_inf, mask_ratio_sup,
                                                len(entropy_sorted))

                for i, entropy in enumerate(self.shannon_entropy[batch_index]):
                    index = entropy_sorted.index(entropy)
                    ratio_list[i] = np.max(mask_ratio_linear[index] * self.basic_mask_ratio, 0)
                self.dynamic_mask_ratio.append(ratio_list)
        else:
            for batch_index in range(len(self.shannon_entropy)):
                ratio_list = np.ones_like(np.asarray(self.shannon_entropy[batch_index], dtype=np.float_)) * self.basic_mask_ratio
                self.dynamic_mask_ratio.append(ratio_list)

    def update_next_step_mask_ratio(self):
        if self.hierarchical_masking is not None:
            self.dynamic_mask_ratio = [x * self.hierarchical_masking for x in self.dynamic_mask_ratio]
        else:
            self.dynamic_mask_ratio = self.dynamic_mask_ratio

    def mask_token(self, x):
        # token resolution
        b_, _, _, _, w_ = x.size()

        for batch_index in range(b_):
            mask_list = []
            if self.grid_resolution[batch_index] > w_:
                pad = self.grid_resolution[batch_index] % w_ // 2
            else:
                pad = w_ % self.grid_resolution[batch_index] // 2

            if pad != 0:
                padding = nn.ConstantPad3d(int(pad), 0.)
                img = padding(x[batch_index].unsqueeze(0)).squeeze(0)
            else:
                img = x[batch_index]

            n, d, h, w = img.size()

            gr = self.grid_resolution[batch_index]
            dmr = self.dynamic_mask_ratio[batch_index]

            img = img.view(n, gr, d // gr, gr, h // gr, gr, w // gr)
            img = img.permute(0, 1, 3, 5, 2, 4, 6)

            # [n, grid_num, token_num_in_grid]
            img = rearrange(img, 'n res_d res_h res_w d h w -> n (res_d res_h res_w) (d h w)')

            for grid_index, mask_ratio in enumerate(dmr):
                num_mask = int(img.size()[-1] * np.min([mask_ratio, 1]))
                mask_seq = np.concatenate([np.zeros(num_mask),
                                           np.ones(img.size()[-1] - num_mask)])

                np.random.shuffle(mask_seq)
                mask_list.append(mask_seq.tolist())

            mask_list = np.array(mask_list)
            img *= torch.from_numpy(mask_list).to(img.device)

            img = img.reshape(n, gr, gr, gr, -1)
            img = img.reshape(n, gr, gr, gr, d // gr, h // gr, w // gr)
            img = img.permute(0, 1, 4, 2, 5, 3, 6)
            img = rearrange(img, 'n gz nz gh nh gw nw -> n (gz nz) (gh nh) (gw nw)')

            if pad != 0:
                img = img[:, pad:-pad, pad:-pad, pad:-pad]

            x[batch_index] = img

            if self.step == 0:
                self.mask_list.append(mask_list)
        return x

    def reset_step(self):
        self.step = 0
        # [b]
        self.grid_resolution = []
        # [b, grid_num]
        self.shannon_entropy = []
        # [b, grid_num]
        self.dynamic_mask_ratio = []

    def get_masked_gt(self, x):
        b, c, d, h, w = x.size()

        # patchy
        x = x.view(b, c, d // 2, 2, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
        x = rearrange(x, 'b c res_z res_h res_w z h w -> b c res_z res_h res_w (z h w)')

        for batch_index in range(b):
            image = x[batch_index]
            _, d_, h_, w_, _ = image.size()

            grid = self.grid_resolution[batch_index]

            image = image.view(c, grid, d_ // grid, grid, h_ // grid, grid, w_ // grid, 8)
            image = image.permute(0, 7, 1, 3, 5, 2, 4, 6)
            image = rearrange(image, 'c n res_z res_h res_w z h w -> c n (res_z res_h res_w) (z h w)')

            image *= (1. - torch.from_numpy(self.mask_list[batch_index])).to(image.device)

            image = image.permute(0, 2, 3, 1)
            image = image.reshape(c, grid, grid, grid, d_ // grid, h_ // grid, w_ // grid, 8)
            image = image.permute(0, 1, 4, 2, 5, 3, 6, 7)
            image = rearrange(image, 'c res_z z res_h h res_w w n -> c (res_z z) (res_h h) (res_w w) n')

            x[batch_index] = image

        x = x.view(b, c, d // 2, h // 2, w // 2, 2, 2, 2).permute(0, 1, 2, 5, 3, 6, 4, 7)
        x = rearrange(x, 'b c res_z z res_h h res_w w -> b c (res_z z) (res_h h) (res_w w)')
        return x







