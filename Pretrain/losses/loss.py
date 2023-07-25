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

import torch
from torch.nn.functional import normalize
import numpy as np


class CrossCorrelationLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, z_i, z_j):
        # empirical cross-correlation matrix
        b, n, c = z_i.size()
        loss = torch.zeros(1).cuda()
        for batch_index in range(b):
            c = z_i[batch_index].T @ z_j[batch_index]

            c.div_(n)

            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = self.off_diagonal(c).pow_(2).sum()
            loss += (on_diag + self.args.lambd * off_diag)
        return loss / b

    @staticmethod
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Loss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.recon_loss = torch.nn.MSELoss().cuda()
        self.contrast_loss = CrossCorrelationLoss(args).cuda()
        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2

    def __call__(self, output_contrastive, target_contrastive, output_recons, target_recons):
        contrast_loss = self.alpha1 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha2 * self.recon_loss(output_recons, target_recons)
        total_loss = contrast_loss + recon_loss
        return total_loss, contrast_loss, recon_loss


if __name__ == '__main__':
    pass