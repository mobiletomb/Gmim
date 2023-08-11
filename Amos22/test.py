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
from functools import partial

import nibabel as nib
import numpy as np
import torch
from dataset.dataloader import get_loader

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import AsDiscrete

from monai.metrics import DiceMetric, compute_surface_dice, SurfaceDistanceMetric
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from utils.utils import AverageMeter, distributed_all_gather
from monai.utils.enums import MetricReduction


parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.7, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=192, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=192, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--pos", default=1, type=int)
parser.add_argument("--neg", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")

parser.add_argument("--fold", default=1, type=int)

parser.add_argument("--datasets", default="BraTsDatasetMonai")
parser.add_argument("--data_dir", default="/dataset/brats2021/", type=str, help="dataset directory")
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--device", default='cuda', type=str)

parser.add_argument("--json_list", default="./jsons/brats21_folds.json", type=str, help="dataset json file")
parser.add_argument("--frac", default=1, type=int)
parser.add_argument( "--output_dir")
parser.add_argument("--pretrained_dir")
parser.add_argument("--best_model")
parser.add_argument("--save_seg", action='store_true')

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
    
    for i in range(12):
        seg_[seg==i] = cmap(i)
        label_[label==i] = cmap(i)

    cmap = plt.cm.Pastel1

    seg_[seg==12] = cmap(2)
    label_[label==12] = cmap(2)
    
    seg_[seg==13] = cmap(7)
    label_[label==13] = cmap(7)
    
    seg_[seg==14] = cmap(5)
    label_[label==14] = cmap(5)
    
    seg_[seg==15] = cmap(6)
    label_[label==15] = cmap(6)
        
    seg = seg_
    label = label_
    
    os.makedirs(output_dir, exist_ok=True)

    
    for i in range(0, d, 5):
        seg_name = os.path.join(output_dir, f'seg_{i}.png')
        label_name = os.path.join(output_dir, f'label_{i}.png')

        plt.imsave(seg_name, seg[:, :, i], dpi=150)
        plt.imsave(label_name, label[:, :, i], dpi=150)

def test(args):
    args.test_mode = True
    
    dsc_csv_path = os.path.join(args.output_dir, f'{str(args.fold)}_dsc.csv')
    nsd_csv_path = os.path.join(args.output_dir, f'{str(args.fold)}_nsd.csv')
    seg_path = os.path.join(args.output_dir, str(args.fold))

    os.makedirs(seg_path, exist_ok=True)

    test_loader = get_loader("Amos2022Dataset",
                                  args.json_list,
                                  args.data_dir,
                                  args.fold,
                                  1,
                                  args.workers,
                                  args,
                                  phase='test')

    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)

    Dice = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    Nsd = SurfaceDistanceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    
    Dice_acc = AverageMeter()
    Nsd_acc = AverageMeter()
    
    model = SwinUNETR(
        img_size=128,
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
    model.to(args.device)

    model_inferer = partial(
        sliding_window_inference,
        roi_size=[128, 128, 128],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )

    SementicCls = ["spleen", "right kidney", "left kidney",
                    "gall bladder", "esophagus", "liver",
                    "stomach", "aorta", "postcava", "pancreas",
                    "right adrenal gland", "left adrenal gland",
                    "duodenum", "bladder", "prostate/uterus"]
    
    results_csv_dsc = {
        'Id': [],
        'Mean Dice': [],
    }
    
    results_csv_nsd = {
        'Id': [],
        'Mean NSD': [],
    }
    
    for cls in range(15):
        results_csv_dsc[f'DSC {SementicCls[cls]}'] = []
        results_csv_nsd[f'NSD {SementicCls[cls]}'] = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            start_time = time()
            image = batch["image"].to(args.device)
            label = batch["label"].to(args.device)
            path = batch['path']

            img_name = path[0].split("/")[-1].split("_")[1][:-7]
            results_csv_dsc['Id'].append(img_name)
            results_csv_nsd['Id'].append(img_name)
            print("Inference on case {}".format(img_name))
            logits = model_inferer(image)
            infer_time = time()
            
            print("Inference complete:", np.round(infer_time-start_time, 4))
            
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in label]
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in logits]
            
            Dice.reset()
            Dice(y_pred=val_output_convert, y=val_labels_convert)
            dice_acc, not_nans = Dice.aggregate()
            dice_acc = dice_acc.cuda()

            Dice_acc.update(dice_acc.cpu().numpy(), n=not_nans.cpu().numpy())
            
            Nsd.reset()
            Nsd(y_pred=val_output_convert, y=val_labels_convert)
            nsd_acc, not_nans = Nsd.aggregate()
            nsd_acc = nsd_acc.cuda()

            Nsd_acc.update(nsd_acc.cpu().numpy(), n=not_nans.cpu().numpy())
            
            end_time = time()
            print(f"Save image to {path} | Metric Time: {np.round(end_time-infer_time, 4)}")

        dice_acc = Dice_acc.avg
        nsd_acc = Nsd_acc.avg
        
        for i, c in enumerate(SementicCls):
            print(f"Dsc {c}:", round(dice_acc[i + 1], 3))
            results_csv_dsc[f'DSC {SementicCls[i]}'].append(round(dice_acc[i + 1], 3)) 

        for i, c in enumerate(SementicCls):
            print(f"Nsd {c}:", round(nsd_acc[i + 1], 3))
            results_csv_nsd[f'NSD {SementicCls[i]}'].append(round(nsd_acc[i + 1], 3)) 

        results_csv_dsc['Mean Dice'].append(np.mean(dice_acc)) 
        results_csv_nsd['Mean NSD'].append(np.mean(nsd_acc))

        csv = pd.DataFrame(results_csv_dsc, index=results_csv_dsc['Id'])
        csv.to_csv(dsc_csv_path, index=False)
        
        csv = pd.DataFrame(results_csv_nsd, index=results_csv_nsd['Id'])
        csv.to_csv(nsd_csv_path, index=False)

        print("Finished inference!")


if __name__ == "__main__":
    args = parser.parse_args()
    test(args)
