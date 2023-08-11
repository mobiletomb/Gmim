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

import argparse
import os
import shutil
import json
from functools import partial

import numpy as np
import torch
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete
from monai.metrics import DiceMetric, HausdorffDistanceMetric, compute_percent_hausdorff_distance

from dataset.dataloader import get_loader

from time import time


parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=4, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")

parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")

parser.add_argument("--datasets", default="BraTsDataset")
parser.add_argument("--data_dir", default="/dataset/brats2021/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="./jsons/brats21_folds.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=1, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--to_origin_classes", action='store_true')
parser.add_argument("--save_seg", action='store_true')
parser.add_argument("--frac", default=1, type=int)
parser.add_argument("--output_dir", default="./", type=str, help="pretrained checkpoint directory")
parser.add_argument("--pretrained_dir")
parser.add_argument("--best_model")
parser.add_argument("--device", default='cpu', )


def visual(label, seg, output_dir):
    _, _, d = label.shape
    cmap = plt.cm.Paired
    
    # 将类别对应到指定的颜色，如果需要修改透明背景时，将每一个颜色的最后一维设置为 0
    seg_ = np.zeros((seg.shape[0], 
                     seg.shape[1],
                     seg.shape[2], 
                     4)) 
    label_ = np.zeros((seg.shape[0], 
                     seg.shape[1],
                     seg.shape[2], 
                     4))

    for i, j in enumerate([0, 2, 1, 4]):
        seg_[seg==j] = cmap(i)
        label_[label==j] = cmap(i)
        
    seg = seg_
    label = label_
    
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, d, 5):

        seg_name = os.path.join(output_dir, f'seg_{i}.png')
        label_name = os.path.join(output_dir, f'label_{i}.png')
        plt.imsave(seg_name, seg[:, :, i], cmap='Paired', dpi=150)
        plt.imsave(label_name, label[:, :, i], cmap='Paired', dpi=150)


def test(args):
    args.test_mode = True
    
    csv_path = os.path.join(args.output_dir, f'{str(args.fold)}.csv')
    seg_path = os.path.join(args.output_dir, str(args.fold))

    os.makedirs(seg_path, exist_ok=True)
        
    print('Test model:', os.path.join(args.pretrained_dir, args.best_model))
    print('Save results to:', args.output_dir)

    test_loader = get_loader(args.datasets,
                                  args.json_list,
                                  args.data_dir,
                                  args.fold,
                                  1,
                                  args.workers,
                                  args,
                                  phase='test')

    post_sigmoid = Activations(sigmoid=True)

    Dice = DiceMetric(include_background=True)

    device = args.device

    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )

    model_path = os.path.join(args.pretrained_dir, args.best_model)

    model_dict = torch.load(model_path, map_location='cpu')["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=[128, 128, 128],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
        device=device,
    )

    results_csv = {
        'Id': [],
        'Dice TC': [],
        'Dice WT': [],
        'Dice ET': [],
        'Haus TC': [],
        'Haus WT': [],
        'Haus ET': [],
    }


    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch["image"].to(device)
            label = batch["label"].to('cpu')
            path = batch['path']

            num = path[0].split("/")[-1].split("_")[1]
            img_name = "BraTS2021_" + num + ".nii.gz"

            print("Inference on case {}".format(img_name))
            start_time = time()
            logits = model_inferer(image)
            seg_out = logits.detach().cpu()

            seg_out = post_sigmoid(seg_out) > 0.5 
            seg_out = seg_out.to(torch.bool)

            dice = Dice(seg_out, label)[0].get_array()

            haus_tc = compute_percent_hausdorff_distance(
                seg_out[0][0].unsqueeze(0).unsqueeze(0),
                label[0][0].unsqueeze(0).unsqueeze(0),
                percentile=99,
            )

            haus_wt = compute_percent_hausdorff_distance(
                seg_out[0][1].unsqueeze(0).unsqueeze(0),
                label[0][1].unsqueeze(0).unsqueeze(0),
                percentile=99,
            )
                        
            haus_et = compute_percent_hausdorff_distance(
                seg_out[0][2].unsqueeze(0).unsqueeze(0),
                label[0][2].unsqueeze(0).unsqueeze(0),
                percentile=99,
            )
            
            haus_tc_ = compute_percent_hausdorff_distance(
                label[0][0].unsqueeze(0).unsqueeze(0),
                seg_out[0][0].unsqueeze(0).unsqueeze(0),
                percentile=99,
            )

            haus_wt_ = compute_percent_hausdorff_distance(
                label[0][1].unsqueeze(0).unsqueeze(0),
                seg_out[0][1].unsqueeze(0).unsqueeze(0),
                percentile=99,
            )
                        
            haus_et_ = compute_percent_hausdorff_distance(
                label[0][2].unsqueeze(0).unsqueeze(0),
                seg_out[0][2].unsqueeze(0).unsqueeze(0),
                percentile=99,
            )
            
            dice = np.round(dice, 4)
            haus_tc = np.round(np.mean([haus_tc, haus_tc_]), 2)
            haus_wt = np.round(np.mean([haus_wt, haus_wt_]), 2)
            haus_et = np.round(np.mean([haus_et, haus_et_]), 2)

            results_csv['Id'].append(num)
            results_csv['Dice TC'].append(dice[0])
            results_csv['Dice WT'].append(dice[1])
            results_csv['Dice ET'].append(dice[2])
            results_csv['Haus TC'].append(haus_tc)
            results_csv['Haus WT'].append(haus_wt)
            results_csv['Haus ET'].append(haus_et)
            
            end_time = time()
            print('Time cost: %.2f' %(end_time - start_time))
            print('DSC:', dice)
            print('Haus:', [haus_tc, haus_wt, haus_et])

            seg_out = seg_out.detach().cpu().numpy()

            seg_out = seg_out[0]
            seg_out_ = np.zeros_like(seg_out[0]).astype(np.uint8)

            seg_out_[seg_out[1] == 1] = 2
            seg_out_[seg_out[0] == 1] = 1
            seg_out_[seg_out[2] == 1] = 4
                
            label = label[0]
            label_ = np.zeros_like(label[0]).astype(np.uint8)
            
            label_[label[1] == 1] = 2
            label_[label[0] == 1] = 1
            label_[label[2] == 1] = 4
        
            path = os.path.join(seg_path, img_name)
            print(f"Save image to {path}")
            visual(label=label_, seg=seg_out_, output_dir=path)

        csv = pd.DataFrame(results_csv, index=results_csv['Id'])
        csv.to_csv(csv_path, index=False)

        print("Finished inference!")


if __name__ == "__main__":
    args = parser.parse_args()
    test(args)
