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

from monai import transforms


transform = transforms.Compose(
    [
        transforms.RandAdjustContrast(prob=0.7, gamma=[0.7, 1.5]),
        transforms.RandGaussianSmooth(prob=0.6, sigma_x=(0.50, 1.00)),
        transforms.RandShiftIntensity(0.4, prob=0.4),
        transforms.RandGaussianNoise(prob=0.5, mean=0.05, std=0.05),
    ]
)


def aug_rand(samples):
    b = samples.size()[0]
    for i in range(b):
        samples[i] = transform(samples[i].unsqueeze(0)).squeeze(0)
    return samples


if __name__ == '__main__':
    import torch
    inps = torch.randn((1, 3, 128, 128, 128))
    out = aug_rand(inps)
    print(out.shape)